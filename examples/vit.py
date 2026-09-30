"""A Vision Transformer trained with domain- *and* tensor parallelism.

The mesh has two axes and every tensor is split over both:

    image   (B, C / col, H, W / row)     channels over "col", width over "row"
    tokens  (B, N / row, E / col)        sequence over "row", embedding over "col"

Nothing is ever gathered: the patch embedding, ring attention, the MLP and the
patch recovery all consume and produce these shards.

    patch embed   DistributedConv2D          kernel == stride, so no halo exchange
    attention     DistributedAttention       ring attention over "row", heads over "col"
    MLP           DistributedXWT             weights split over both axes
    layer norm    DistributedLayerNorm       statistics reduced over "col"
    recovery      DistributedConv2DTranspose

Attention is global: every token attends to the whole image, not just to the tokens
on its own rank -- the ring passes the key/value chunks around "row".

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/vit.py
"""

import torch
import torch.nn as nn
from _common import setup, shard

import jigsaw
from jigsaw import log
from jigsaw.modules.attention.config import RingAttentionConfig, SendForward
from jigsaw.utils import set_all_seeds

device, rank, world_size = setup()

mesh = jigsaw.make_process_mesh({"row": 2, "col": 2})
row, col, model_group = mesh["row"], mesh["col"], mesh["model"]

# Every rank needs its own seed so that the shards are initialised independently.
set_all_seeds(model_group.rank())

IN_CHANNELS, PATCH, DIM, HEADS, DEPTH = 4, 4, 64, 4, 2
H, W = 32, 64  # global image size


class Block(nn.Module):
    """Pre-norm transformer block: ``x + attn(norm(x))``, then ``x + mlp(norm(x))``."""

    def __init__(self) -> None:
        super().__init__()

        def norm() -> nn.Module:
            return jigsaw.normalization.DistributedLayerNorm(
                local_dim=DIM // col.size(),  # each rank normalises its slice of E ...
                local_rank=model_group.rank(),
                device=device,
                row_group=row,
                col_group=col,  # ... and the statistics are reduced over "col"
            )

        def linear(in_dim: int, out_dim: int) -> nn.Module:
            return jigsaw.linear.DistributedXWT(
                in_dim,
                out_dim,
                bias=True,
                local_rank=model_group.rank(),
                device=device,
                model_group=model_group,  # the weight is split over row *and* col
                row_group=row,
                column_group=col,
            )

        self.norm1, self.norm2 = norm(), norm()
        self.attn = jigsaw.attention.DistributedAttention(
            in_dim=DIM,
            embed_dim=DIM,
            out_dim=DIM,
            heads=HEADS,
            dropout_p=0.0,
            local_rank=model_group.rank(),
            device=device,
            spatial_group=row,  # the sequence is split over "row" ...
            channel_group=col,  # ... the heads over "col"
            ring=True,  # every token attends to all tokens
            config=RingAttentionConfig(
                device=device,
                # The ring passes the keys/values around. Sending them (rather than
                # the default, the input x) is required with a channel split: x is
                # only a slice of the features, too little to rebuild K and V from.
                send_in_forward=SendForward.KV,
                # Ring ranks per node. The default, 4, needs a ring of at least 4.
                gpus_per_node=row.size(),
            ),
        )
        self.mlp = nn.Sequential(linear(DIM, 4 * DIM), nn.GELU(), linear(4 * DIM, DIM))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def sincos_pos_embed(
    dim: int, h_idx: torch.Tensor, w_idx: torch.Tensor
) -> torch.Tensor:
    """Fixed 2-D sin/cos embedding evaluated at *global* patch indices.

    Because it is a function of the global position, a token gets the same embedding
    no matter which rank it lives on -- so the model does not depend on the mesh.
    """
    omega = 1.0 / 10000 ** (torch.arange(dim // 4) / (dim // 4))

    def axis(idx: torch.Tensor) -> torch.Tensor:
        angle = idx[:, None] * omega[None]
        return torch.cat([angle.sin(), angle.cos()], dim=-1)  # (n, dim / 2)

    h, w = axis(h_idx), axis(w_idx)
    return torch.cat(
        [
            h[:, None].expand(-1, len(w_idx), -1),
            w[None].expand(len(h_idx), -1, -1),
        ],
        dim=-1,
    )  # (Hp, Wp, dim)


class ViT(nn.Module):
    """Patch embedding -> position embedding -> transformer blocks -> patch recovery."""

    def __init__(self) -> None:
        super().__init__()
        conv_kwargs = dict(
            kernel_size=PATCH,
            stride=PATCH,  # non-overlapping patches: no halo exchange needed
            device=device,
            spatial_group=row,  # a single group splits the last (width) axis
            channel_group=col,
        )
        self.embed = jigsaw.convolution.DistributedConv2D(
            IN_CHANNELS, DIM, **conv_kwargs
        )
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = jigsaw.normalization.DistributedLayerNorm(
            DIM // col.size(), model_group.rank(), device, row, col
        )
        self.recover = jigsaw.convolution.DistributedConv2DTranspose(
            DIM, IN_CHANNELS, **conv_kwargs
        )

        # This rank owns patch columns [w0, w0 + wp) and embedding channels [c0, c0 + e).
        hp, wp, e = H // PATCH, W // PATCH // row.size(), DIM // col.size()
        pos = sincos_pos_embed(
            DIM,
            torch.arange(hp),
            row.rank() * wp + torch.arange(wp),
        )[..., col.rank() * e : (col.rank() + 1) * e]
        self.register_buffer("pos_embed", pos.to(device), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)  # (B, E / col, Hp, Wp / row)
        b, e, hp, wp = z.shape
        z = z.permute(0, 2, 3, 1) + self.pos_embed  # (B, Hp, Wp / row, E / col)
        # tokens: (B, N / row, E / col). The ring sends its input to the neighbour
        # ranks, which needs contiguous memory (the permute above left it strided).
        z = z.reshape(b, hp * wp, e).contiguous()
        for block in self.blocks:
            z = block(z)
        z = self.norm(z)
        # Going back to an image, the gradient of the token tensor arrives strided
        # (it is a permuted view); the distributed linears want it contiguous.
        z.register_hook(lambda grad: grad.contiguous())
        z = z.reshape(b, hp, wp, e).permute(0, 3, 1, 2)
        return x + self.recover(z)  # predict a correction to the input


model = ViT()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = jigsaw.loss.DistributedMSELoss(reduce_groups=(row, col))
log.info("local parameters per rank: %d", jigsaw.utils.count_parameters(model))


def make_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's shard of a noisy input and its clean target."""
    g = torch.Generator().manual_seed(step)
    clean = torch.nn.functional.avg_pool2d(
        torch.randn(4, IN_CHANNELS, H, W, generator=g), 5, 1, 2
    )
    noisy = clean + 0.3 * torch.randn(clean.shape, generator=g)

    def local(x: torch.Tensor) -> torch.Tensor:
        return shard(shard(x, 1, col), -1, row).to(device)

    return local(noisy), local(clean)


for step in range(100):
    x, y = make_batch(step)  # each (4, C / col, H, W / row)
    loss = loss_fn(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        log.info("step %3d  loss %.4f", step, loss.item())
