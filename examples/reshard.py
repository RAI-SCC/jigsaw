"""Move a trained model to a different process mesh, or gather it into plain PyTorch.

A model is trained on a mesh with model parallelism (here 2 x 2 = 4 ranks per replica)
because it is too big, or too slow, for fewer. For inference you often want something
else: fewer ranks per replica and more replicas. The layers can re-shard their own
parameters from one mesh onto another:

    layer.reshard_parameters(from_mesh, to_mesh)

which gathers the weights and keeps the slice the layer needs on the new mesh, rebinding
the layer to the new groups. Both meshes must span the same ranks. Here we go from

    training   {"row": 2, "col": 2}             one replica on all 4 ranks
    inference  {"dp": 4, "row": 1, "col": 1}    4 replicas, each rank holds the full model

For a single, plain PyTorch copy -- to save a checkpoint that needs no jigsaw, say --
``layer.convert_to_torch(mesh)`` merges the shards into an ``nn.Linear`` / ``nn.Conv2d``.

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/reshard.py
"""

import torch
import torch.nn as nn
from _common import setup, shard

import jigsaw
from jigsaw import log
from jigsaw.utils import set_all_seeds
from jigsaw.utils_aggregate import gather_results

DIM = 64

def main(steps: int = 100) -> None:
    """Train on a 2 x 2 mesh, then export to plain PyTorch and re-shard for inference."""
    device, rank, world_size = setup()

    # --- train on a 2 x 2 mesh: weights are split over both axes -----------------------
    train_mesh = jigsaw.make_process_mesh({"row": 2, "col": 2})
    row, col = train_mesh["row"], train_mesh["col"]
    set_all_seeds(train_mesh["model"].rank())  # independent shards, so different seeds

    def linear(in_dim: int, out_dim: int) -> nn.Module:
        return jigsaw.linear.DistributedXWT(
            in_dim,
            out_dim,
            local_rank=train_mesh["model"].rank(),
            device=device,
            model_group=train_mesh["model"],
            column_group=col,
            row_group=row,
            bias=True,
        )

    # tokens are split over "row" (sequence) and "col" (features): (B, S / row, DIM / col)
    model = nn.Sequential(linear(DIM, 4 * DIM), nn.GELU(), linear(4 * DIM, DIM))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = jigsaw.loss.DistributedMSELoss(reduce_groups=(row, col))

    def local(x: torch.Tensor) -> torch.Tensor:
        """This rank's shard of a full (B, S, DIM) tensor on the training mesh."""
        return shard(shard(x, 1, row), 2, col).to(device)

    for step in range(steps):
        x = torch.randn(4, 8, DIM, generator=torch.Generator().manual_seed(step))
        loss = loss_fn(model(local(x)), local(torch.tanh(x)))  # learn tanh
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    log.info("trained: loss %.4f", loss.item())

    x_test = torch.randn(2, 8, DIM, generator=torch.Generator().manual_seed(1234))
    with torch.no_grad():  # the trained model's answer, stitched back together
        expected = gather_results(model(local(x_test)), train_mesh)

    # --- export to plain PyTorch: rank 0 saves a checkpoint that needs no jigsaw -------
    # `convert_to_torch` is collective (it gathers), so every rank calls it.
    plain = nn.Sequential(
        model[0].convert_to_torch(train_mesh),
        nn.GELU(),
        model[2].convert_to_torch(train_mesh),
    )
    with torch.no_grad():
        err = (plain(x_test.to(device)) - expected).abs().max()
    log.info("plain nn.Sequential vs distributed: max diff %.2e", err)
    assert err < 1e-4, err
    # if rank == 0:
    #     torch.save(plain.state_dict(), "model.pt")  # loads with plain PyTorch, anywhere

    # --- or re-shard onto another mesh: one full replica per rank ----------------------
    infer_mesh = jigsaw.make_process_mesh(
        {"dp": 4, "row": 1, "col": 1}, model_parallel_axes=("row", "col")
    )
    for layer in model.modules():
        if hasattr(layer, "reshard_parameters"):  # every layer that owns parameters
            layer.reshard_parameters(train_mesh, infer_mesh)

    # On this mesh nothing is split, so the model takes full, unsharded inputs -- and
    # every rank can serve its own requests.
    with torch.no_grad():
        out = model(x_test.to(device))
    err = (out - expected).abs().max()
    log.info("resharded onto 4 replicas: max diff %.2e", err)
    assert err < 1e-4, err


if __name__ == "__main__":
    main()
