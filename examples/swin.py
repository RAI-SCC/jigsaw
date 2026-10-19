"""A Swin Transformer trained with domain parallelism: window attention needs no ring.

Swin attends inside small non-overlapping windows and alternates two layouts:

    W-MSA    windows on a fixed grid
    SW-MSA   the same grid, after the feature map is shifted by half a window

Cut the feature map into a 2x2 grid of tiles, one tile per rank, and choose the tile
size to be a multiple of the window size. Then every window lies inside one tile and
almost everything is local:

    window attention    no communication -- a batch of small independent attentions
    patch merging       no communication -- a 2x2 neighbourhood never spans two tiles
    MLP, layer norm     no communication -- they act on one token at a time
    shift               ``DistributedRoll`` -- the only layer that talks to neighbours

The shift is a cyclic ``torch.roll`` of the *whole* map, so a few rows and columns have
to cross the tile borders. ``DistributedRoll`` does exactly that: a halo exchange with
the neighbouring tiles, wrapping around the border of the image.

Two details that are easy to get wrong with a split image:

  * The mask of the shifted windows (tokens that were wrapped around the image border
    must not attend to each other) depends on the *global* position of a window, so
    each rank builds it from the position of its tile.
  * The weights are replicated, and ``jigsaw`` only synchronises the ones inside its own
    layers. ``replicate`` below does the same for the plain PyTorch layers: broadcast
    the weights from rank 0 and mark them replicated over the tiles; ``Swin.forward``
    then routes them through ``jigsaw.parallel.synced_parameters``, which sums their
    gradients over the tiles inside the backward pass.

Tokens are channels-last, ``(B, H / sh, W / sw, C)``; the convolutions that embed and
recover the patches are channels-first, as in PyTorch. This is a denoiser, so it ends
with a transposed convolution rather than a classification head.

Run with one process per GPU (or 4 CPU processes):

    torchrun --nproc-per-node 4 examples/swin.py

Data parallelism on top (``--ddp``, a multiple of 4 ranks): a leading "dp" axis holds
copies of the tiled model, each training on its own images.

    torchrun --nproc-per-node 8 examples/swin.py --ddp

Every weight is replicated, so each gradient is summed over the tiles (in the backward
pass, by the mark ``replicate`` sets) and then averaged over "dp" by a plain
``DistributedDataParallel`` over ``mesh["dp"]``. ``replicate`` broadcasts the weights
over the tiles; DDP only broadcasts over "dp". If you mark parameters after building
the model, ``jigsaw.parallel.wrap_ddp`` broadcasts them for you. With ``--joint-hook``
the sum and the average become one all-reduce in DDP's buckets (see ``docs/ddp.md``).
"""

import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from _common import data_seed, ddp_flags, setup, shard, steps_flag

import jigsaw
from jigsaw import log, parallel

IN_CHANNELS, PATCH, WINDOW = 1, 2, 4
DIMS, HEADS, DEPTHS = (32, 64), (2, 4), (2, 2)  # per stage
H, W = 64, 64  # global image size

Mesh = dict[str, dist.ProcessGroup]


def replicate(module: nn.Module, mesh: Mesh, device: torch.device) -> nn.Module:
    """Give every rank the same copy of ``module`` on ``device``, and keep them identical.

    Every rank sees only its tile, so its gradient is only a part of the total. Summing
    the parts over the tiles gives every rank the gradient of the whole image; the
    mark tells ``jigsaw.parallel`` to do that sum (see ``Swin.forward``).
    """
    module.to(device)  # NCCL only broadcasts GPU tensors
    for p in module.parameters():
        dist.broadcast(p.data, group_src=0, group=mesh["model"])
        parallel.mark_replicated(p, mesh["model"])
    return module


def windows(x: torch.Tensor) -> torch.Tensor:
    """Cut a tile ``(B, h, w, C)`` into windows ``(B, nW, WINDOW * WINDOW, C)``."""
    b, h, w, c = x.shape
    x = x.view(b, h // WINDOW, WINDOW, w // WINDOW, WINDOW, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, -1, WINDOW * WINDOW, c)


def unwindows(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Inverse of :func:`windows`."""
    b, c = x.shape[0], x.shape[-1]
    x = x.view(b, h // WINDOW, w // WINDOW, WINDOW, WINDOW, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def relative_position_index() -> torch.Tensor:
    """Index into the bias table for every pair of positions inside a window."""
    ij = torch.stack(torch.meshgrid(*[torch.arange(WINDOW)] * 2, indexing="ij"))
    ij = ij.flatten(1)  # (2, N)
    rel = ij[:, :, None] - ij[:, None, :] + (WINDOW - 1)  # (2, N, N), in [0, 2W - 2]
    return rel[0] * (2 * WINDOW - 1) + rel[1]


def shift_mask(h: int, w: int, shift: int, mesh: Mesh) -> torch.Tensor:
    """Mask ``(nW, N, N)`` of the windows of this tile after rolling the map by -shift.

    Rolling moves the top rows and left columns of the *image* to its bottom and right.
    Inside a window, the tokens from the two sides of the image border must not attend
    to each other, so give every token the id of the region it came from and let tokens
    attend to those with the same id. The border lies at the *global* position, so the
    region of a token follows from where this rank's tile sits in the token grid.
    """

    def region(local_size: int, group: dist.ProcessGroup) -> torch.Tensor:
        size = local_size * group.size()  # global size of the token grid
        idx = group.rank() * local_size + torch.arange(local_size)
        return (idx >= size - WINDOW).long() + (idx >= size - shift).long()

    rows, cols = region(h, mesh["sh"]), region(w, mesh["sw"])
    ids = (rows[:, None] * 3 + cols[None, :])[None, :, :, None]  # (1, h, w, 1)
    ids = windows(ids).squeeze(-1).squeeze(0)  # (nW, N)
    same = ids[:, :, None] == ids[:, None, :]
    return torch.zeros(same.shape).masked_fill(~same, float("-inf"))


class WindowAttention(nn.Module):
    """Multi-head attention inside windows, with a learned relative position bias."""

    def __init__(
        self,
        dim: int,
        heads: int,
        tile: tuple[int, int],
        shift: int,
        mesh: Mesh,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.bias_table = nn.Parameter(torch.zeros((2 * WINDOW - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        replicate(self, mesh, device)

        self.register_buffer(
            "index", relative_position_index().to(device), persistent=False
        )
        if shift:
            mask = shift_mask(*tile, shift, mesh).to(device)
            self.register_buffer("mask", mask, persistent=False)
        else:
            self.mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x.shape
        x = windows(x)  # (B, nW, N, C): the windows are just a batch dimension
        n = x.shape[2]
        qkv = self.qkv(x).view(b, -1, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(3, 0, 1, 4, 2, 5).unbind(0)  # (B, nW, heads, N, d)

        bias = self.bias_table[self.index.view(-1)].view(n, n, -1).permute(2, 0, 1)
        if self.mask is not None:
            bias = bias[None] + self.mask[:, None]  # (nW, heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(2, 3).reshape(b, -1, n, c)
        return unwindows(self.proj(out), h, w)


class SwinBlock(nn.Module):
    """``x + attn(norm(x))``, then ``x + mlp(norm(x))``; the windows may be shifted."""

    def __init__(
        self,
        dim: int,
        heads: int,
        tile: tuple[int, int],
        mesh: Mesh,
        device: torch.device,
        shift: int = 0,
    ) -> None:
        super().__init__()
        self.norm1 = replicate(nn.LayerNorm(dim), mesh, device)
        self.attn = WindowAttention(dim, heads, tile, shift, mesh, device)
        self.norm2 = replicate(nn.LayerNorm(dim), mesh, device)
        self.mlp = replicate(
            nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim)),
            mesh,
            device,
        )

        def roll(shift: int) -> nn.Module:
            return jigsaw.modules.DistributedRoll(
                shifts=(shift, shift),
                dims=(1, 2),  # the two spatial axes of (B, H, W, C) ...
                distributed_dims=(1, 2),  # ... both are split ...
                spatial_groups=(mesh["sh"], mesh["sw"]),  # ... over "sh" and "sw"
                device=device,
            )

        # Shifting by -shift moves the windows; the opposite roll puts the result back.
        self.shift = roll(-shift) if shift else nn.Identity()
        self.unshift = roll(shift) if shift else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.unshift(self.attn(self.shift(self.norm1(x))))
        return x + self.mlp(self.norm2(x))


class PatchMerging(nn.Module):
    """Halve the resolution: concatenate every 2x2 group of tokens, project to 2C."""

    def __init__(self, dim: int, mesh: Mesh, device: torch.device) -> None:
        super().__init__()
        self.norm = replicate(nn.LayerNorm(4 * dim), mesh, device)
        self.reduce = replicate(nn.Linear(4 * dim, 2 * dim, bias=False), mesh, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # A tile has an even size and starts on an even row and column, so its 2x2
        # groups are exactly those of the undivided map: no communication.
        x = torch.cat(
            [x[:, 0::2, 0::2], x[:, 1::2, 0::2], x[:, 0::2, 1::2], x[:, 1::2, 1::2]],
            dim=-1,
        )
        return self.reduce(self.norm(x))


class Swin(nn.Module):
    """Patch embedding -> Swin stages joined by patch merging -> patch recovery."""

    def __init__(self, mesh: Mesh, device: torch.device) -> None:
        super().__init__()
        sh, sw = mesh["sh"], mesh["sw"]
        conv_kwargs = dict(
            device=device, spatial_group=(sh, sw), channel_group=mesh["col"]
        )
        # kernel == stride: the patches do not overlap, so there is no halo to exchange.
        self.embed = jigsaw.modules.DistributedConv2d(
            IN_CHANNELS, DIMS[0], kernel_size=PATCH, stride=PATCH, **conv_kwargs
        )

        tile = (H // PATCH // sh.size(), W // PATCH // sw.size())  # local token grid
        stages: list[nn.Module] = []
        for i, (dim, heads, depth) in enumerate(zip(DIMS, HEADS, DEPTHS)):
            # The tile must hold a whole number of windows, and the shift a window.
            assert tile[0] % WINDOW == 0 and tile[1] % WINDOW == 0, tile
            blocks = [
                SwinBlock(
                    dim,
                    heads,
                    tile,
                    mesh,
                    device,
                    shift=0 if j % 2 == 0 else WINDOW // 2,
                )
                for j in range(depth)
            ]
            if i < len(DIMS) - 1:
                blocks.append(PatchMerging(dim, mesh, device))
                tile = (tile[0] // 2, tile[1] // 2)
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        scale = PATCH * 2 ** (len(DIMS) - 1)  # pixels per token of the last stage
        self.norm = replicate(nn.LayerNorm(DIMS[-1]), mesh, device)
        self.recover = jigsaw.modules.DistributedConvTranspose2d(
            DIMS[-1], IN_CHANNELS, kernel_size=scale, stride=scale, **conv_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run the model with the replicated weights swapped for copies whose gradient
        # is summed over the tiles in the backward pass. The copies carry no mark, so
        # the re-entered forward finds nothing left to sync and runs the layers.
        synced = parallel.synced_parameters(self)
        if synced:
            return torch.func.functional_call(self, synced, (x,))
        z = self.embed(x).permute(0, 2, 3, 1)  # channels-last: (B, H', W', C)
        z = self.norm(self.stages(z))
        z = z.permute(0, 3, 1, 2).contiguous()  # back to channels-first
        return x + self.recover(z)  # predict a correction to the input


def make_batch(
    step: int, mesh: Mesh, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return this rank's tile of a noisy input and its clean target.

    Every rank of a model copy draws the same images (and keeps its tile of them);
    every data-parallel copy draws different ones.
    """
    g = torch.Generator().manual_seed(data_seed(step, mesh))
    clean = F.avg_pool2d(torch.randn(4, IN_CHANNELS, H, W, generator=g), 5, 1, 2)
    noisy = clean + 0.3 * torch.randn(clean.shape, generator=g)

    def tile(x: torch.Tensor) -> torch.Tensor:
        return shard(shard(x, -2, mesh["sh"]), -1, mesh["sw"]).to(device)

    return tile(noisy), tile(clean)


def main(
    steps: int = 100, data_parallel: bool = False, joint_hook: bool = False
) -> list[float]:
    """Train the Swin denoiser for ``steps`` steps and return the loss of every step.

    With ``data_parallel`` the losses are those of this rank's model copy.
    """
    device, rank, world_size = setup()

    if data_parallel:
        # The tiles of one copy are its "model" group; "dp" counts the copies.
        mesh = jigsaw.make_process_mesh(
            {"dp": -1, "sh": 2, "sw": 2, "col": 1},
            model_parallel_axes=("sh", "sw", "col"),
        )
    else:
        mesh = jigsaw.make_process_mesh({"sh": 2, "sw": 2, "col": 1})

    model = Swin(mesh, device)
    if data_parallel:
        # DDP over "dp" only, checked right away (wrong group, unequal replicas).
        model = jigsaw.parallel.wrap_ddp(model, mesh, joint_hook=joint_hook)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = jigsaw.modules.DistributedMSELoss(reduce_groups=(mesh["sh"], mesh["sw"]))
    log.info(
        "parameters: %d per rank (all replicated), %d in the model",
        jigsaw.debug.count_parameters(model),
        jigsaw.debug.count_parameters(model, mesh),
    )

    losses = []
    for step in range(steps):
        x, y = make_batch(step, mesh, device)  # each (4, 1, H / 2, W / 2)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 20 == 0:
            log.info("step %3d  loss %.4f", step, losses[-1])
    return losses


if __name__ == "__main__":
    main(**ddp_flags(sys.argv), **steps_flag(sys.argv))
