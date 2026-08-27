import gc
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

    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

    config.addinivalue_line("markers", "no_dist_env: skip the dist_env autouse fixture")
    config.addinivalue_line("markers", "gpu_required: test requires GPU")
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
def dist_env(device: torch.device) -> Generator[None, None, None]:
    """Initialize the distributed process group once for the entire test session."""
    if device.type == "cuda":
        torch.cuda.set_device(device)
        assert dist.is_nccl_available()
    else:
        assert dist.is_gloo_available()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

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

    for item in items:
        if item.get_closest_marker("gpu_required") and not has_gpu:
            item.add_marker(pytest.mark.skip(reason="GPU not available"))
        if item.get_closest_marker("dist_required") and not is_distributed:
            item.add_marker(
                pytest.mark.skip(reason="Not running in distributed environment")
            )
