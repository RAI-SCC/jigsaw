"""Jigsaw benchmarking harness.

Not shipped in the wheel (see ``pyproject.toml``). Reproducibly times the
forward+backward cost of jigsaw modules on dummy data, records a hardware
fingerprint alongside every number, and diffs a run against a committed
per-hardware baseline.

Entry point: ``python -m benchmarks <run|compare|promote> ...``.
"""
