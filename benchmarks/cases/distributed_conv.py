"""Distributed N-D convolution: grouped vs full-channel, on a row x col mesh.

Times ``DistributedConv2D`` / ``DistributedConv3D`` from
:mod:`jigsaw.modules.convolution`. One step is ``conv(x).sum().backward()`` on
this rank's shard (channels split over ``col``, last spatial axis over ``row``).

Vary ``--groups`` to compare regimes at a fixed global problem size:

* ``groups=1``           -- full channel gather (one path, balanced).
* ``groups=col.size()``  -- channel-local grouped conv, **no** channel-group
  communication (balanced).
* ``groups`` not ``1`` and not a multiple of ``col.size()`` -- rejected with
  ``NotImplementedError`` today; recorded as *skipped* so this case starts
  producing the imbalance comparison the moment non-divisible groups land.

    srun -n4 scripts/run.sh python -m benchmarks run distributed_conv --mesh 1x4 --dim 3 --groups 1 4

Reported TFLOP/s is an approximate **global** ``6 * MACs`` over the slowest-rank
wall time; ``imbalance`` is ``max/mean`` of the per-rank medians.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from benchmarks.harness import launch, mesh
from benchmarks.harness.driver import run_grid
from jigsaw.modules.convolution import DistributedConv2D, DistributedConv3D

_CONV = {2: DistributedConv2D, 3: DistributedConv3D}
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _flops(batch: int, cin: int, cout: int, groups: int, kernel, spatial) -> int:
    macs = batch * cout * (cin // groups) * math.prod(kernel) * math.prod(spatial)
    return int(6 * macs)  # 2 MAC->FLOP, ~3x for fwd+bwd


def _builder(ctx: launch.LaunchCtx, md: dict, args: argparse.Namespace):
    rows, cols = md["row"].size(), md["col"].size()

    def build(params: dict):
        groups, channel_comm = params["groups"], params["channel_comm"]
        dim = args.dim
        kernel = tuple([args.kernel] * dim)
        spatial = tuple([args.spatial] * dim)
        cin, cout, batch = args.in_channels, args.out_channels, args.batch

        if cin % cols or cout % cols:
            raise NotImplementedError(
                f"channels {cin}/{cout} not divisible by col={cols}"
            )
        if spatial[-1] % rows:
            raise NotImplementedError(
                f"spatial {spatial[-1]} not divisible by row={rows}"
            )
        if cin % groups or cout % groups:
            raise NotImplementedError(f"channels not divisible by groups={groups}")

        dt = _DTYPES[params["dtype"]]
        conv = _CONV[dim](
            cin,
            cout,
            kernel,
            device=ctx.device,
            spatial_group=md["row"],
            channel_group=md["col"],
            stride=1,
            padding=tuple(k // 2 for k in kernel),
            dilation=1,
            groups=groups,
            bias=True,
            dtype=dt,
            channel_comm=channel_comm,
        ).to(ctx.device)

        spatial_local = list(spatial)
        spatial_local[-1] //= rows
        x = torch.randn(
            batch,
            cin // cols,
            *spatial_local,
            device=ctx.device,
            dtype=dt,
            requires_grad=True,
        )

        def step() -> None:
            conv(x).sum().backward()

        def reset() -> None:
            conv.zero_grad(set_to_none=True)
            x.grad = None

        return step, reset, _flops(batch, cin, cout, groups, kernel, spatial), None

    return build


def main(argv: list[str] | None = None) -> int:
    """Bring up the mesh, run the groups x channel-comm grid, write the run."""
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks run distributed_conv",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--mesh", default="1x1", help="row x col, e.g. 1x4 (default 1x1)")
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--in-channels", type=int, default=256)
    ap.add_argument("--out-channels", type=int, default=256)
    ap.add_argument(
        "--spatial", type=int, default=128, help="size of every spatial axis"
    )
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--groups", type=int, nargs="+", default=[1])
    ap.add_argument(
        "--channel-comm",
        nargs="+",
        default=["auto"],
        choices=["auto", "all_gather", "p2p", "all_to_all"],
    )
    ap.add_argument("--dtypes", nargs="+", default=["float32"], choices=list(_DTYPES))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=15.0)
    ap.add_argument("--min-block-ms", type=float, default=50.0)
    args = ap.parse_args(argv)

    ctx = launch.init(args.device)
    torch.manual_seed(args.seed + ctx.rank)

    md = mesh.make_mesh(ctx.device.type, mesh.parse_mesh(args.mesh))
    knobs = {
        "seed": args.seed,
        "mesh": args.mesh,
        "dim": args.dim,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "spatial": args.spatial,
        "kernel": args.kernel,
        "batch": args.batch,
        "device": ctx.device.type,
        "backend": ctx.backend,
        "min_block_ms": args.min_block_ms,
        "time_budget_s": args.time_budget_s,
    }
    combos = [
        {"groups": g, "channel_comm": cc, "dtype": d}
        for d in args.dtypes
        for g in args.groups
        for cc in args.channel_comm
    ]
    try:
        run_grid(
            case="distributed_conv",
            combos=combos,
            build=_builder(ctx, md, args),
            device=ctx.device,
            group=md["model"],
            world_size=ctx.world_size,
            knobs=knobs,
            out_dir=args.out,
            file_tag=f"{args.mesh}-d{args.dim}",
            min_block_ms=args.min_block_ms,
            time_budget_s=args.time_budget_s,
        )
    finally:
        launch.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
