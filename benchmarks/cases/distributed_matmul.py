"""Distributed parameter-free matmul on a row x col mesh.

Times ``DistributedMatMul{XW,XWT,XTW}`` from
:mod:`jigsaw.modules.linear.linear` -- the sharded ``a @ b`` / ``a @ b.mT`` /
``a.mT @ b`` with no learnable weights. One step is ``model(a, b).sum().backward()``.

    # single node, 4 GPUs, 2x2 mesh
    srun -n4 scripts/run.sh python -m benchmarks run distributed_matmul --mesh 2x2 --variants xw xwt

Operands are square ``n x n`` globally; each rank holds an ``(n/row, n/col)``
shard (the layout ``tests/modules/linear/test_matmul.py`` verifies). Reported
TFLOP/s is the **global** ``6 n^3`` over the slowest-rank wall time; the
``imbalance`` column is ``max/mean`` of the per-rank medians.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from benchmarks.harness import launch, mesh
from benchmarks.harness.driver import run_grid
from jigsaw.modules.linear.linear import (
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
)

_VARIANTS = {
    "xw": DistributedMatMulXW,
    "xwt": DistributedMatMulXWT,
    "xtw": DistributedMatMulXTW,
}
_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _builder(ctx: launch.LaunchCtx, md: dict):
    rows, cols = md["row"].size(), md["col"].size()

    def build(params: dict):
        n, variant, dtype = params["n"], params["variant"], params["dtype"]
        if n % rows or n % cols:
            raise NotImplementedError(f"n={n} not divisible by mesh {rows}x{cols}")
        model = _VARIANTS[variant](
            local_rank=md["model"].rank(),
            device=ctx.device,
            parallelism=md["model"].size(),
            model_group=md["model"],
        )
        dt = _DTYPES[dtype]
        a = torch.randn(
            n // rows, n // cols, device=ctx.device, dtype=dt, requires_grad=True
        )
        b = torch.randn(
            n // rows, n // cols, device=ctx.device, dtype=dt, requires_grad=True
        )

        def step() -> None:
            model(a, b).sum().backward()

        def reset() -> None:
            a.grad = None
            b.grad = None

        return step, reset, 6 * n**3, None

    return build


def main(argv: list[str] | None = None) -> int:
    """Bring up the mesh, run the variant x size x dtype grid, write the run."""
    ap = argparse.ArgumentParser(
        prog="python -m benchmarks run distributed_matmul",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("--mesh", default="1x1", help="row x col, e.g. 2x2 (default 1x1)")
    ap.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--variants", nargs="+", default=["xw"], choices=list(_VARIANTS))
    ap.add_argument("--dtypes", nargs="+", default=["bfloat16"], choices=list(_DTYPES))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time-budget-s", type=float, default=15.0)
    ap.add_argument("--min-block-ms", type=float, default=50.0)
    args = ap.parse_args(argv)

    ctx = launch.init(args.device)
    torch.manual_seed(args.seed + ctx.rank)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    md = mesh.make_mesh(ctx.device.type, mesh.parse_mesh(args.mesh))
    knobs = {
        "seed": args.seed,
        "mesh": args.mesh,
        "allow_tf32": args.tf32,
        "device": ctx.device.type,
        "backend": ctx.backend,
        "min_block_ms": args.min_block_ms,
        "time_budget_s": args.time_budget_s,
    }
    combos = [
        {"variant": v, "n": n, "dtype": d}
        for v in args.variants
        for d in args.dtypes
        for n in args.sizes
    ]
    try:
        run_grid(
            case="distributed_matmul",
            combos=combos,
            build=_builder(ctx, md),
            device=ctx.device,
            group=md["model"],
            world_size=ctx.world_size,
            knobs=knobs,
            out_dir=args.out,
            file_tag=args.mesh,
            min_block_ms=args.min_block_ms,
            time_budget_s=args.time_budget_s,
        )
    finally:
        launch.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
