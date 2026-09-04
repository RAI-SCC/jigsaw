"""Unit tests for ``jigsaw.utils.resolve_launch_env``.

``resolve_launch_env`` is the pure environment-parsing half of the launch
contract that ``scripts/run.sh`` sets up: it turns ``srun`` / ``torchrun`` /
bare-shell environment variables into a :class:`~jigsaw.utils.LaunchEnv`. It
touches no CUDA and starts no process group, so these are plain unit tests --
every launcher environment is faked with ``monkeypatch``.
"""

import dataclasses
import os

import pytest

from jigsaw.utils import LaunchEnv, resolve_launch_env

# Every variable ``resolve_launch_env`` reads. The autouse fixture clears all of
# them so a test starts from a known-empty environment regardless of whether the
# suite itself is running under srun.
_LAUNCH_VARS = (
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
    "SLURM_JOB_NODELIST",
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


@pytest.fixture(autouse=True)
def _clean_launch_env(monkeypatch):
    """Remove every launcher variable; ``monkeypatch`` restores them after."""
    for name in _LAUNCH_VARS:
        monkeypatch.delenv(name, raising=False)


def _fake_scontrol(hostnames_output, recorder=None):
    """Return a stand-in for ``subprocess.check_output`` yielding fixed bytes."""

    def _check_output(cmd, *args, **kwargs):
        if recorder is not None:
            recorder.append(cmd)
        return hostnames_output

    return _check_output


class TestSource:
    """Which launcher ``resolve_launch_env`` believes it is running under."""

    def test_slurm(self, monkeypatch):
        """``SLURM_PROCID`` present -> the SLURM triple is read verbatim."""
        monkeypatch.setenv("SLURM_PROCID", "2")
        monkeypatch.setenv("SLURM_NTASKS", "8")
        monkeypatch.setenv("SLURM_LOCALID", "1")

        env = resolve_launch_env()

        assert env.source == "slurm"
        assert (env.rank, env.world_size, env.local_rank) == (2, 8, 1)

    def test_torchrun(self, monkeypatch):
        """No ``SLURM_PROCID`` but ``RANK`` present -> the torchrun triple."""
        monkeypatch.setenv("RANK", "3")
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "2")

        env = resolve_launch_env()

        assert env.source == "torchrun"
        assert (env.rank, env.world_size, env.local_rank) == (3, 4, 2)

    def test_single_fallback(self):
        """Neither launcher -> a well-defined single-process result."""
        env = resolve_launch_env()

        assert env.source == "single"
        assert (env.rank, env.world_size, env.local_rank) == (0, 1, 0)

    def test_slurm_wins_when_both_present(self, monkeypatch):
        """srun sets ``RANK`` too; ``SLURM_PROCID`` must take precedence."""
        monkeypatch.setenv("SLURM_PROCID", "5")
        monkeypatch.setenv("SLURM_NTASKS", "16")
        monkeypatch.setenv("RANK", "0")
        monkeypatch.setenv("WORLD_SIZE", "1")

        env = resolve_launch_env()

        assert env.source == "slurm"
        assert (env.rank, env.world_size) == (5, 16)


class TestStrict:
    """``strict=True`` refuses to guess a single-process run."""

    def test_raises_without_launcher(self):
        with pytest.raises(RuntimeError, match="No distributed launcher detected"):
            resolve_launch_env(strict=True)

    def test_ok_with_launcher(self, monkeypatch):
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "1")

        env = resolve_launch_env(strict=True)

        assert env.source == "slurm"


class TestDefaults:
    """Missing optional variables fall back the way the docstring promises."""

    def test_slurm_missing_ntasks_and_localid(self, monkeypatch):
        """Only ``SLURM_PROCID`` -> world_size 1, local_rank 0."""
        monkeypatch.setenv("SLURM_PROCID", "0")

        env = resolve_launch_env()

        assert (env.world_size, env.local_rank) == (1, 0)

    def test_torchrun_missing_world_size_and_local_rank(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")

        env = resolve_launch_env()

        assert (env.world_size, env.local_rank) == (1, 0)

    def test_blank_slurm_procid_raises(self, monkeypatch):
        """A set-but-empty ``SLURM_PROCID`` is a sharp edge.

        ``os.getenv("SLURM_PROCID") is not None`` is ``True`` for ``""``, so the
        SLURM branch is taken and ``int("")`` raises. Documents current
        behaviour -- arguably it should fall through to the next launcher.
        """
        monkeypatch.setenv("SLURM_PROCID", "")

        with pytest.raises(ValueError):
            resolve_launch_env()


class TestMasterAddr:
    """``MASTER_ADDR``: explicit env > SLURM nodelist via scontrol > localhost."""

    def test_explicit_env_wins_and_skips_scontrol(self, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "10.0.0.5")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "node[01-04]")

        calls: list = []
        monkeypatch.setattr(
            "jigsaw.utils.subprocess.check_output",
            _fake_scontrol(b"node01\n", recorder=calls),
        )

        env = resolve_launch_env()

        assert env.master_addr == "10.0.0.5"
        assert calls == [], "scontrol must not run when MASTER_ADDR is already set"

    def test_derived_from_nodelist(self, monkeypatch):
        """No ``MASTER_ADDR`` -> first host of ``scontrol show hostnames``."""
        monkeypatch.setenv("SLURM_PROCID", "0")
        monkeypatch.setenv("SLURM_NTASKS", "4")
        monkeypatch.setenv("SLURM_JOB_NODELIST", "node[01-04]")

        calls: list = []
        monkeypatch.setattr(
            "jigsaw.utils.subprocess.check_output",
            _fake_scontrol(b"node01\nnode02\nnode03\nnode04\n", recorder=calls),
        )

        env = resolve_launch_env()

        assert env.master_addr == "node01"
        assert calls == [["scontrol", "show", "hostnames", "node[01-04]"]]

    def test_localhost_when_no_nodelist(self):
        env = resolve_launch_env()

        assert env.master_addr == "localhost"


class TestMasterPort:
    def test_default(self):
        assert resolve_launch_env().master_port == "29500"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("MASTER_PORT", "12321")

        assert resolve_launch_env().master_port == "12321"


class TestEnvironWriteBack:
    """The resolved rendezvous values land back in ``os.environ`` for ``env://``."""

    def test_explicit_values_written_back(self, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "10.1.2.3")
        monkeypatch.setenv("MASTER_PORT", "27000")

        env = resolve_launch_env()

        assert os.environ["MASTER_ADDR"] == "10.1.2.3" == env.master_addr
        assert os.environ["MASTER_PORT"] == "27000" == env.master_port

    def test_defaults_written_back(self):
        env = resolve_launch_env()

        assert os.environ["MASTER_ADDR"] == "localhost" == env.master_addr
        assert os.environ["MASTER_PORT"] == "29500" == env.master_port


def test_returns_launch_env_instance():
    assert isinstance(resolve_launch_env(), LaunchEnv)


def test_launch_env_is_frozen():
    """``LaunchEnv`` is an immutable value object."""
    env = resolve_launch_env()

    with pytest.raises(dataclasses.FrozenInstanceError):
        env.rank = 99  # type: ignore[misc]
