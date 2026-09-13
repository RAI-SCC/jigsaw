"""Command-line entry point: ``python -m benchmarks <run|compare|promote> ...``."""

from __future__ import annotations

import importlib
import sys

_CASES = {
    "calibration_matmul": "benchmarks.cases.calibration_matmul",
    "distributed_matmul": "benchmarks.cases.distributed_matmul",
    "distributed_conv": "benchmarks.cases.distributed_conv",
}

_USAGE = """usage:
  python -m benchmarks run <case> [case args]      run a benchmark case
  python -m benchmarks compare <run.jsonl> [...]   diff a run vs its baseline
  python -m benchmarks promote <run.jsonl> [...]   make a run the baseline

cases: {cases}
""".format(cases=", ".join(_CASES))


def main(argv: list[str]) -> int:
    """Dispatch ``run`` / ``compare`` / ``promote`` and return an exit code."""
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]

    if cmd == "run":
        if not rest or rest[0] not in _CASES:
            print(_USAGE)
            return 2
        mod = importlib.import_module(_CASES[rest[0]])
        return mod.main(rest[1:])

    if cmd in ("compare", "promote"):
        from benchmarks.harness import compare

        return compare.main(argv)

    print(_USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
