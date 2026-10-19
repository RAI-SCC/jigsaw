import gc
import importlib.util
import os
import sys
import warnings
from collections.abc import Generator
from typing import Any

import mpi4py.MPI as MPI
import pytest
import torch
import torch.distributed as dist

from tests.process_mesh import *  # noqa: F401

comm = MPI.COMM_WORLD
mpi_size = comm.Get_size()
mpi_rank = comm.Get_rank()


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register project-specific pytest command-line options.

    Parameters
    ----------
    parser : pytest.Parser
        Argument parser to configure.

    """
    parser.addoption(
        "--device",
        help="Device to run tests on (cpu or cuda)",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Also run tests marked slow (the full run that gates a release)",
    )

    parser.addoption(
        "--print-all-ranks",
        action="store_true",
        default=False,
        help="Allow all ranks to print; by default only rank 0 prints",
    )


def pytest_configure(config):
    """Configure rank 0 to be the only printing rank."""
    if mpi_rank != 0 and not config.getoption("--print-all-ranks"):
        config.option.verbose = -1
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    # The reference-comparison tests check a distributed op against a
    # single-process ``F.*`` result with ~1e-4 tolerances. On Ampere+ GPUs cuDNN
    # / cuBLAS run fp32 convs and matmuls in TF32 by default (~1e-3 relative
    # error), and which kernel a call lands on depends on its shape -- so the
    # sharded path and the whole-tensor reference disagree well past 1e-4 even
    # though the maths is identical. There is no TF32 on CPU, which is why the
    # same tests pass there. Force true fp32 so the tolerances mean something.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    config.addinivalue_line("markers", "no_dist_env: skip the dist_env autouse fixture")
    config.addinivalue_line(
        "markers", "slow: expensive extra coverage; only runs with --runslow"
    )
    config.addinivalue_line("markers", "gpu_required: test requires GPU")
    config.addinivalue_line(
        "markers", "te_required: test requires transformer_engine (container path)"
    )
    config.addinivalue_line(
        "markers", "dist_required: test requires distributed environment"
    )
    config.addinivalue_line(
        "markers",
        "hybrid_dist_env: test requires distributed environment on both CPU and GPU.",
    )
    config.addinivalue_line(
        "markers",
        "dist(n_ranks): Mark test to run only when running parallel tests, and optionaly, only with the specified ranks",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    # Check if the test has the 'dist' marker or the 'gpu' marker, and skip if the conditions are not met
    """Apply test-specific skip conditions before execution.

    Parameters
    ----------
    item : pytest.Item
        Test item being configured.

    """
    gpu_marker = item.get_closest_marker("gpu")
    if gpu_marker and (
        not torch.cuda.is_available()
        or item.config.getoption("--device") not in ["cuda", "gpu"]
    ):
        pytest.skip("Skipping GPU test since CUDA is not available or not selected.")

    skip_marker = item.get_closest_marker("dist")
    if skip_marker:
        ranks = int(skip_marker.args[0]) if len(skip_marker.args) == 1 else None
        if mpi_size <= 1 or (ranks is not None and ranks != mpi_size):
            pytest.skip(
                f"Skipping distributed test, only {mpi_size} ranks available, but {ranks if ranks else '> 1'} required"
            )


@pytest.fixture(autouse=True, scope="session")
def dist_env(
    request: pytest.FixtureRequest, device: torch.device
) -> Generator[None, None, None]:
    """Initialize the shared process group once for the whole test session.

    Stays session-scoped and autouse so it runs before the session-scoped
    ``process_mesh_dict_*`` fixtures, which need the default group. When *every*
    selected test is marked ``no_dist_env`` -- e.g. running ``test_gloo_nccl.py``
    on its own, which brings up its own ``cpu:gloo,cuda:nccl`` group -- the
    shared group is not created, so that is not a double init.
    """
    items = request.session.items
    if items and all(
        item.get_closest_marker("no_dist_env") is not None for item in items
    ):
        yield None
        return

    if device.type == "cuda":
        torch.cuda.set_device(device)
        assert dist.is_nccl_available()
    else:
        assert dist.is_gloo_available()

    # Default to a single-machine rendezvous for local/CI runs; a real launcher
    # that already computed the job's actual rendezvous host (e.g. run.sh
    # deriving it from SLURM_JOB_NODELIST) exports MASTER_ADDR itself, and that
    # must win here or a multi-node run can never actually rendezvous --
    # "localhost" resolves to a different loopback on every node.
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")

    dist.init_process_group(
        backend="nccl" if device.type == "cuda" else "gloo",
        rank=mpi_rank,
        world_size=mpi_size,
        device_id=device.index if device.type == "cuda" else None,
    )
    if not dist.is_initialized():
        raise RuntimeError("Process group initialization failed.")
    yield None
    dist.destroy_process_group()


@pytest.fixture(scope="session")
def device(request: Any) -> torch.device:
    """Return the device selected for the test session.

    Parameters
    ----------
    request : Any
        Pytest request object providing test configuration.

    Returns
    -------
    torch.device
        Result returned by the operation.
    """
    device = request.config.getoption("--device")
    torch.set_default_device("cpu")
    if device == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu")

    else:
        local_id = mpi_rank % torch.cuda.device_count()
        device = torch.device("cuda", local_id)
        return device


@pytest.fixture(autouse=True)
def restore_torch_numerics() -> Generator[None, None, None]:
    """Undo a test's changes to torch's global numerics settings.

    Some tests switch on ``torch.use_deterministic_algorithms`` for a reproducible
    reference. Left on, it makes every later cuBLAS matmul on the GPU raise (unless
    ``CUBLAS_WORKSPACE_CONFIG`` is set), failing unrelated tests further down the run.
    """
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    yield
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


@pytest.fixture(autouse=False)
def cuda_cleanup():
    """Release CUDA resources after a test."""
    yield
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()


def pytest_collection_modifyitems(config, items):
    """Skip GPU/distributed tests based on environment."""
    has_gpu = torch.cuda.is_available()
    is_distributed = (
        os.environ.get("SLURM_PROCID") is not None
        or os.environ.get("RANK") is not None
        or mpi_size > 1
    )
    has_te = importlib.util.find_spec("transformer_engine") is not None
    run_slow = config.getoption("--runslow")

    for item in items:
        if item.get_closest_marker("slow") and not run_slow:
            item.add_marker(pytest.mark.skip(reason="slow: run with --runslow"))
        if item.get_closest_marker("gpu_required") and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        if item.get_closest_marker("dist_required") and not is_distributed:
            item.add_marker(
                pytest.mark.skip(reason="Not running in distributed environment")
            )
        if item.get_closest_marker("te_required") and not has_te:
            item.add_marker(pytest.mark.skip(reason="transformer_engine not installed"))
