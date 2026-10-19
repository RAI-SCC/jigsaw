"""The tutorials the way ``examples/README.md`` tells a user to run them.

``test_examples.py`` imports the tutorials and runs them inside the test session. This
file starts them as a user would -- ``torchrun --nproc-per-node 4 examples/x.py`` -- in
a job of its own, so it runs in the single-process CI job and is skipped under ``mpirun``.
It always uses the CPU (gloo), whatever the machine has.
"""

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import mpi4py.MPI as MPI
import pytest

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() > 1,
    reason="starts its own 4-process job: run it without mpirun",
)

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
SCRIPTS = sorted(p.stem for p in EXAMPLES.glob("*.py") if not p.name.startswith("_"))

# Anything a launcher (this one, mpirun, srun) put in the environment would be
# inherited by the job we start and be taken for *its* launch.
_LAUNCHER_VARS = (
    "SLURM_",
    "OMPI_",
    "PMI_",
    "PMIX_",
    "MASTER_",
    "TORCHELASTIC_",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
)


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(_LAUNCHER_VARS)}
    env["CUDA_VISIBLE_DEVICES"] = ""  # CPU / gloo
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO), *filter(None, [env.get("PYTHONPATH")])]
    )
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_scripts_found() -> None:
    assert len(SCRIPTS) >= 6, SCRIPTS


@pytest.mark.parametrize("name", SCRIPTS)
def test_runs_under_torchrun(name: str) -> None:
    """``torchrun --nproc-per-node 4 examples/<name>.py`` finishes without an error."""
    proc = _run(
        [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            f"examples/{name}.py",
        ]
    )
    assert proc.returncode == 0, f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"


def test_bare_python_says_how_to_launch() -> None:
    """The first thing a user tries -- ``python examples/x.py`` -- explains itself."""
    proc = _run(["examples/domain_parallel.py"])
    assert proc.returncode != 0
    assert "needs exactly 4 ranks" in proc.stderr, proc.stderr[-2000:]
    assert "launch with 4 ranks" in proc.stderr


def test_every_script_is_in_the_readme() -> None:
    """The table of ``examples/README.md`` lists exactly the scripts there are."""
    readme = (EXAMPLES / "README.md").read_text()
    listed = re.findall(r"^\| \[`(\w+)\.py`\]", readme, flags=re.MULTILINE)
    assert sorted(listed) == SCRIPTS


@pytest.mark.parametrize("name", SCRIPTS)
def test_script_is_importable_and_guarded(name: str) -> None:
    """Scripts define ``main()`` and only run it as ``__main__``; the tests rely on it."""
    source = (EXAMPLES / f"{name}.py").read_text()
    assert "def main(" in source
    assert 'if __name__ == "__main__":' in source


def _load_common(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``examples/_common.py``, the helpers the tutorials share."""
    monkeypatch.syspath_prepend(str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "jigsaw_example_common", EXAMPLES / "_common.py"
    )
    assert spec is not None and spec.loader is not None
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    return common


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        ([], (False, False)),
        (["--ddp"], (True, False)),
        (["--joint-hook"], (True, True)),  # the hook needs DDP: it implies --ddp
        (["--ddp", "--joint-hook"], (True, True)),
    ],
)
def test_ddp_flags(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: tuple[bool, bool]
) -> None:
    """``--ddp`` and ``--joint-hook`` as the tutorials' ``main()`` arguments."""
    flags = _load_common(monkeypatch).ddp_flags(["examples/vit.py", *argv])
    assert (flags["data_parallel"], flags["joint_hook"]) == expected


@pytest.mark.parametrize(
    ("env", "world_size", "expected"),
    [
        ({}, 4, 4),  # a plain launch: everything is on this node
        ({"LOCAL_WORLD_SIZE": "4"}, 8, 4),  # torchrun on two nodes
        ({"SLURM_NTASKS_PER_NODE": "2"}, 8, 2),  # srun
    ],
)
def test_processes_on_this_node(
    monkeypatch: pytest.MonkeyPatch, env: dict, world_size: int, expected: int
) -> None:
    """``setup()`` compares this against the GPU count to choose between NCCL and CPU."""
    for var in ("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    assert _load_common(monkeypatch)._processes_on_this_node(world_size) == expected


@pytest.mark.parametrize(
    ("visible", "env", "expected"),
    [
        (4, {}, 4),  # torchrun: every process sees the node's GPUs
        (0, {}, 0),  # no GPU
        (1, {}, 1),  # one GPU for all the processes, as on a laptop
        # srun binding one GPU to each task: one is visible, the step has four
        (1, {"SLURM_PROCID": "2", "SLURM_STEP_GPUS": "0,1,2,3"}, 4),
        (1, {"SLURM_PROCID": "0", "SLURM_STEP_GPUS": "0"}, 1),  # a one-GPU step
        (4, {"SLURM_PROCID": "2", "SLURM_STEP_GPUS": "0,1,2,3"}, 4),  # srun, unbound
        (0, {"SLURM_PROCID": "2", "SLURM_STEP_GPUS": "0,1,2,3"}, 0),  # no CUDA at all
        (1, {"SLURM_STEP_GPUS": "0,1,2,3"}, 1),  # torchrun inside an allocation
    ],
)
def test_gpus_on_this_node(
    monkeypatch: pytest.MonkeyPatch, visible: int, env: dict, expected: int
) -> None:
    """The GPU count ``setup()`` uses also holds when ``srun`` binds a GPU per task."""
    for var in ("SLURM_PROCID", "SLURM_STEP_GPUS"):
        monkeypatch.delenv(var, raising=False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    common = _load_common(monkeypatch)
    monkeypatch.setattr(common.torch.cuda, "device_count", lambda: visible)
    assert common._gpus_on_this_node() == expected
