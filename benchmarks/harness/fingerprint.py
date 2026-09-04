"""Environment capture for reproducibility.

``collect()`` returns a dict describing the machine, the software stack and the
jigsaw checkout. ``hw_key()`` hashes the parts that must match for two runs to be
comparable -- it names the baseline file a run is checked against.
``read_clocks()`` samples the live GPU clocks / temperature / throttle reasons so
a suspicious number can be spotted after the fact.
"""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]

__all__ = ["collect", "hw_key", "read_clocks"]


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=15)
        return out.decode().strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _pip_freeze_sha256() -> str | None:
    out = _run([sys.executable, "-m", "pip", "freeze"])
    if out is None:
        return None
    return hashlib.sha256(out.encode()).hexdigest()[:16]


def collect(world_size: int = 1) -> dict:
    """Return a fingerprint dict for the current process/environment."""
    import torch

    fp: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": getattr(torch.version, "git_version", None),
        "cuda_runtime": torch.version.cuda,
        "cudnn": _cudnn_version(),
        "cpu": _cpu_model(),
        "world_size": world_size,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "container_sif": os.getenv("JIGSAW_SIF"),
        "jigsaw_commit": _run(["git", "-C", str(_REPO), "rev-parse", "HEAD"]),
        "jigsaw_dirty": bool(_run(["git", "-C", str(_REPO), "status", "--porcelain"])),
        "pip_freeze_sha256": _pip_freeze_sha256(),
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        fp.update(
            gpu_name=props.name,
            gpu_count=torch.cuda.device_count(),
            gpu_capability=f"{props.major}.{props.minor}",
            gpu_mem_gib=round(props.total_memory / 2**30, 1),
            gpu_sm_count=props.multi_processor_count,
            gpu_l2_bytes=getattr(props, "L2_cache_size", None)
            or getattr(props, "l2_cache_size", None),
            driver=_first_line(
                _run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ]
                )
            ),
            nccl=_nccl_version(),
        )
    else:
        fp["gpu_name"] = None
    return fp


def _first_line(text: str | None) -> str | None:
    return text.splitlines()[0].strip() if text else None


def _nccl_version() -> str | None:
    try:
        import torch

        return ".".join(map(str, torch.cuda.nccl.version()))
    except (ImportError, AttributeError, RuntimeError):
        return None


def _cudnn_version() -> int | None:
    try:
        import torch

        if torch.backends.cudnn.is_available():
            return torch.backends.cudnn.version()
    except (ImportError, AttributeError, RuntimeError):
        pass
    return None


def hw_key(fp: dict) -> str:
    """12-hex key over the fields that must match for runs to be comparable."""
    parts = [
        str(fp.get("gpu_name") or "cpu"),
        str(fp.get("driver") or "").split(".")[0],
        str(fp.get("cuda_runtime") or "").split(".")[0],
        str(fp.get("torch") or ""),
        str(fp.get("world_size", 1)),
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def read_clocks() -> dict | None:
    """Sample live GPU clocks / temp / power / throttle reasons (or ``None``)."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,"
            "clocks_throttle_reasons.active",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    if out is None:
        return None
    fields = [f.strip() for f in out.split(",")]
    if len(fields) < 5:
        return {"raw": out}
    return {
        "sm_mhz": _int(fields[0]),
        "mem_mhz": _int(fields[1]),
        "temp_c": _int(fields[2]),
        "power_w": _float(fields[3]),
        "throttle": fields[4],
    }


def _int(s: str) -> int | None:
    try:
        return int(float(s))
    except ValueError:
        return None


def _float(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None
