"""Convert an existing PyTorch model with ``convert_to_jigsaw``.

The model below is ordinary PyTorch: it knows nothing about process meshes.
``jigsaw.convert.convert_to_jigsaw`` walks it and swaps every layer it knows for its
distributed twin, carrying the weights over -- each rank keeps only its shard of them:

    nn.Linear                       -> DistributedXWT
    nn.Conv{1,2,3}d                 -> DistributedConv{1,2,3}D
    nn.ConvTranspose{1,2,3}d        -> DistributedConv{1,2,3}DTranspose
    nn.LayerNorm                    -> DistributedLayerNorm
    jigsaw.attention.Attention      -> DistributedAttention
    jigsaw.roll.Roll                -> DistributedRoll

Everything else (activations, reshapes, ...) is left alone, which is fine as long as it
is elementwise or only touches the *local* shapes. A layer the converter does not know
would silently compute on the shard only if it touches the sharded dimension -- swap
those by hand (see ``vit.py``).

``nn.LayerNorm`` is only converted when its normalized dimension is actually sharded at
that point in the model; pass ``convert_layer_norm=False`` to leave it a plain
``nn.LayerNorm`` instead -- correct only if that dimension is replicated (not split)
there, and worth it to skip the distributed version's extra all-reduces.

The converted model then expects sharded inputs: the mesh's "row" axis splits the
image width (or the token sequence), its "col" axis splits the channels.

We check the conversion the way you should check yours: run the original model on the
full input and compare against the gathered output of the converted one.

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/autoconvert.py
"""

import copy

import torch
import torch.nn as nn
from _common import setup, shard

import jigsaw
from jigsaw import log
from jigsaw.convert import convert_to_jigsaw
from jigsaw.utils import setup_precision
from jigsaw.utils_aggregate import gather_1d_results


class PatchMLP(nn.Module):
    """Patchify an image, mix the tokens with an MLP, and unpatchify. Plain PyTorch."""

    def __init__(self, channels: int = 4, dim: int = 64, patch: int = 4) -> None:
        super().__init__()
        self.embed = nn.Conv2d(channels, dim, kernel_size=patch, stride=patch)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)
        self.recover = nn.ConvTranspose2d(
            dim, channels, kernel_size=patch, stride=patch
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)  # (B, E, Hp, Wp)
        # Only *local* shapes are used below, so this code is valid on a shard too.
        hp, wp = z.shape[-2:]
        z = z.flatten(2).transpose(1, 2).contiguous()  # tokens (B, N, E)
        z = self.fc2(self.act(self.fc1(z)))
        z = z.transpose(1, 2).unflatten(2, (hp, wp))
        return self.recover(z)


def main() -> None:
    """Convert :class:`PatchMLP`, then check forward, backward and the way back."""
    device, rank, world_size = setup()
    # On Ampere and newer GPUs cuDNN runs fp32 convolutions in TF32 by default (~1e-3
    # relative error), and the sharded and the full convolution do not round alike.
    # Compare in true fp32, so that the tolerances below mean something.
    setup_precision(tf32=False)
    mesh = jigsaw.make_process_mesh({"row": 2, "col": 2})
    row, col = mesh["row"], mesh["col"]

    # The same seed on every rank gives every rank the same full model to convert from.
    torch.manual_seed(0)
    reference = PatchMLP().to(device)
    model = convert_to_jigsaw(copy.deepcopy(reference), mesh, device)
    log.info("converted layers: %s", [type(m).__name__ for m in model.children()])

    x = torch.randn(2, 4, 32, 64, device=device)  # same full input on every rank
    x_local = shard(shard(x, 1, col), -1, row)  # channels over "col", width over "row"

    # --- forward: the gathered output equals the original model's output ---
    with torch.no_grad():
        expected = reference(x)
        out = model(x_local)
        out = gather_1d_results(out, mesh, dim=1, pm_key="col")  # channels
        out = gather_1d_results(out, mesh, dim=-1, pm_key="row")  # width
    err = (out - expected).abs().max()
    log.info("forward   max |converted - original| = %.2e", err)
    log.info("(for scale: max |original| = %.2e)", expected.abs().max())
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)

    # --- backward: the gradient of a weight shard equals the same slice of the original
    reference(x).square().mean().backward()
    loss = jigsaw.loss.DistributedMSELoss(reduce_groups=(row, col))
    loss(model(x_local), torch.zeros_like(x_local)).backward()

    # `fc1.weight` is (4E, E): rows split over "row", columns over "col".
    ref_grad = reference.fc1.weight.grad
    n_rows, n_cols = ref_grad.shape[0] // row.size(), ref_grad.shape[1] // col.size()
    ref_shard = ref_grad[
        row.rank() * n_rows : (row.rank() + 1) * n_rows,
        col.rank() * n_cols : (col.rank() + 1) * n_cols,
    ]
    err = (model.fc1.weight.grad - ref_shard).abs().max() / ref_shard.abs().max()
    log.info("backward  max relative error of the fc1.weight.grad shard = %.2e", err)
    assert err < 1e-4, err

    # --- and back: merge the shards into a plain nn.Linear on every rank ---
    plain = model.fc1.convert_to_torch(mesh)
    log.info("fc1 merged back into %s", plain)
    torch.testing.assert_close(plain.weight, reference.fc1.weight)
    torch.testing.assert_close(plain.bias, reference.fc1.bias)


if __name__ == "__main__":
    main()
