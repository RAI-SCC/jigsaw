"""Unit tests for jigsaw.log: rank-targeting, configuration, formatters and collectives."""

import json
import logging
import sys
import warnings

import pytest
import torch.distributed as dist

from jigsaw import log


@pytest.fixture
def records():
    """Capture jigsaw-logger records through the real rank filter."""
    captured: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    logger = log.get_logger()
    handler = _Capture(level=logging.DEBUG)
    handler.addFilter(log._RankFilter())
    old_level, old_propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield captured
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate


@pytest.fixture(autouse=True)
def _reset_state():
    """Restore module globals between tests."""
    log.set_print_rank(0)
    log._scope_ranks = None
    log._seen_once.clear()
    yield
    log.set_print_rank(0)
    log._scope_ranks = None
    log._seen_once.clear()


def _messages(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records]


def test_default_emits_only_from_print_rank(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (0, 4))
    log.info("hello %d", 1)

    monkeypatch.setattr(log, "_rank_world", lambda: (3, 4))
    log.info("suppressed")

    assert _messages(records) == ["hello 1"]


def test_set_print_rank_moves_the_speaker(records, monkeypatch):
    log.set_print_rank(2)
    monkeypatch.setattr(log, "_rank_world", lambda: (2, 4))
    log.info("from 2")
    monkeypatch.setattr(log, "_rank_world", lambda: (0, 4))
    log.info("not from 0 anymore")

    assert _messages(records) == ["from 2"]


def test_at_targets_int_list_and_all(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (2, 4))
    log.at(2).warning("single")
    log.at([0, 1]).warning("excluded")
    log.at("all").warning("everyone")

    assert _messages(records) == ["single", "everyone"]


def test_info_all_ignores_print_rank(records, monkeypatch):
    log.set_print_rank(0)
    monkeypatch.setattr(log, "_rank_world", lambda: (5, 8))
    log.info_all("broadcast")

    assert _messages(records) == ["broadcast"]


def test_only_rank_scopes_the_default(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (1, 4))
    log.set_print_rank(0)

    log.info("before")
    with log.only_rank(1):
        log.info("inside")
    log.info("after")

    assert _messages(records) == ["inside"]


def test_log_once_fires_a_single_time(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
    for _ in range(5):
        log.log_once("only once %s", "x")

    assert _messages(records) == ["only once x"]


def test_records_carry_rank_metadata(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (3, 16))
    log.at("all").info("meta")

    (record,) = records
    assert (record.rank, record.world) == (3, 16)
    assert isinstance(record.elapsed, float)


def test_rank_world_reads_launcher_env_without_process_group(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    for var in ("RANK", "WORLD_SIZE", "SLURM_PROCID", "SLURM_NTASKS"):
        monkeypatch.delenv(var, raising=False)
    assert log._rank_world() == (0, 1)

    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    assert log._rank_world() == (3, 8)

    monkeypatch.setenv("RANK", "5")  # torchrun wins over slurm
    monkeypatch.setenv("WORLD_SIZE", "6")
    assert log._rank_world() == (5, 6)

    monkeypatch.delenv("WORLD_SIZE")
    assert log._rank_world() == (5, 1)


def test_local_rank_prefers_torchrun_over_slurm(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("SLURM_LOCALID", raising=False)
    assert log._local_rank() == 0
    monkeypatch.setenv("SLURM_LOCALID", "2")
    assert log._local_rank() == 2
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert log._local_rank() == 1


def test_resolve_targets_forms():
    assert log._resolve_targets("all") is None
    assert log._resolve_targets(3) == {3}
    assert log._resolve_targets([0, 2]) == {0, 2}
    log.set_print_rank(4)
    assert log._resolve_targets(None) == {4}


@pytest.mark.parametrize(
    "name, expected",
    [
        (None, "jigsaw"),
        ("", "jigsaw"),
        ("jigsaw", "jigsaw"),
        ("jigsaw.attn", "jigsaw.attn"),
        ("attn.ring", "jigsaw.attn.ring"),
    ],
)
def test_get_logger_namespaces_under_jigsaw(name, expected):
    assert log.get_logger(name).name == expected


def test_debug_and_error_and_exception_proxies(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
    log.debug("d")
    log.error("e")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("caught")

    assert _messages(records) == ["d", "e", "caught"]
    assert [r.levelno for r in records] == [logging.DEBUG, logging.ERROR, logging.ERROR]
    assert records[-1].exc_info is not None


def test_disabled_level_is_dropped_before_formatting(records, monkeypatch):
    monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
    log.get_logger().setLevel(logging.ERROR)

    class Exploding:
        def __str__(self):
            raise AssertionError("formatted a disabled record")

    log.info("%s", Exploding())
    assert records == []


def test_only_rank_restores_previous_scope_on_error(records, monkeypatch):
    with pytest.raises(RuntimeError):
        with log.only_rank(1):
            raise RuntimeError
    assert log._scope_ranks is None


class TestFormatters:
    def _record(self, level=logging.INFO, name="jigsaw.sub", exc_info=None):
        record = logging.LogRecord(name, level, __file__, 1, "msg %d", (7,), exc_info)
        record.rank, record.world, record.elapsed = 2, 4, 1.5  # type: ignore[attr-defined]
        return record

    def test_human_plain_is_single_line_with_short_name(self):
        line = log._HumanFormatter(colour=False).format(self._record())
        assert line == "   1.50s [2/4] INFO    sub: msg 7"

    def test_human_root_logger_name_is_dash(self):
        line = log._HumanFormatter(colour=False).format(self._record(name="jigsaw"))
        assert " -: msg 7" in line

    def test_human_colour_wraps_warnings_but_not_info(self):
        fmt = log._HumanFormatter(colour=True)
        warn = fmt.format(self._record(logging.WARNING))
        assert warn.startswith("\033[33m") and warn.endswith("\033[0m")
        assert "\033" not in fmt.format(self._record(logging.INFO))

    def test_human_appends_traceback(self):
        try:
            raise KeyError("k")
        except KeyError:
            exc_info = sys.exc_info()
        text = log._HumanFormatter(colour=False).format(self._record(exc_info=exc_info))
        assert "Traceback" in text and "KeyError" in text

    def test_json_payload(self):
        payload = json.loads(log._JsonFormatter().format(self._record()))
        assert payload == {
            "t": 1.5,
            "rank": 2,
            "world": 4,
            "level": "INFO",
            "name": "jigsaw.sub",
            "msg": "msg 7",
        }

    def test_json_includes_exception(self):
        try:
            raise KeyError("k")
        except KeyError:
            exc_info = sys.exc_info()
        payload = json.loads(
            log._JsonFormatter().format(self._record(exc_info=exc_info))
        )
        assert "KeyError" in payload["exc"]


@pytest.fixture
def isolated_config(monkeypatch):
    """Let a test call ``configure_logging`` and restore all global side effects."""
    logger = log.get_logger()
    saved = (list(logger.handlers), logger.level, logger.propagate)
    saved_hook = sys.excepthook
    saved_state = (log._configured, log._print_rank)
    saved_warnings = warnings.showwarning
    py_warnings = logging.getLogger("py.warnings")
    saved_py = list(py_warnings.handlers)
    for var in (
        "JIGSAW_LOG_LEVEL",
        "JIGSAW_LOG_RANK",
        "JIGSAW_LOG_DIR",
        "JIGSAW_LOG_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)
    log._configured = False
    try:
        yield
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            handler.close()
        for handler in saved[0]:
            logger.addHandler(handler)
        logger.setLevel(saved[1])
        logger.propagate = saved[2]
        for handler in py_warnings.handlers[:]:
            py_warnings.removeHandler(handler)
        for handler in saved_py:
            py_warnings.addHandler(handler)
        logging.captureWarnings(False)
        warnings.showwarning = saved_warnings
        sys.excepthook = saved_hook
        log._configured, log._print_rank = saved_state


class TestConfigureLogging:
    def _kw(self):
        # Emit from every rank so the assertions hold under mpirun too.
        return dict(
            rank=log._rank_world()[0],
            capture_warnings=False,
            install_excepthook=False,
            enable_faulthandler=False,
        )

    def test_human_console_output(self, isolated_config, capsys):
        logger = log.configure_logging("INFO", colour=False, **self._kw())
        log.info("hello %s", "world")
        err = capsys.readouterr().err
        assert "hello world" in err and "INFO" in err
        assert logger.level == logging.INFO and logger.propagate is False

    def test_json_console_output(self, isolated_config, capsys):
        log.configure_logging(as_json=True, **self._kw())
        log.info("as json")
        payload = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
        assert payload["msg"] == "as json"

    def test_second_call_is_a_no_op_unless_forced(self, isolated_config):
        first = log.configure_logging("INFO", **self._kw())
        handlers = list(first.handlers)
        log.configure_logging("DEBUG", **self._kw())
        assert first.level == logging.INFO and first.handlers == handlers
        log.configure_logging("DEBUG", force=True, **self._kw())
        assert first.level == logging.DEBUG and first.handlers != handlers

    def test_environment_supplies_defaults(
        self, isolated_config, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("JIGSAW_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("JIGSAW_LOG_RANK", "0")
        monkeypatch.setenv("JIGSAW_LOG_DIR", str(tmp_path / "logs"))
        monkeypatch.setenv("JIGSAW_LOG_FORMAT", "JSON")
        logger = log.configure_logging(
            capture_warnings=False, install_excepthook=False, enable_faulthandler=False
        )
        assert logger.level == logging.WARNING
        assert log._print_rank == 0
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        assert isinstance(logger.handlers[0].formatter, log._JsonFormatter)

    def test_log_dir_writes_a_per_rank_file(self, isolated_config, tmp_path):
        rank = log._rank_world()[0]
        log.configure_logging("INFO", log_dir=tmp_path / "nested", **self._kw())
        log.info_all("to file")
        for handler in log.get_logger().handlers:
            handler.flush()
        text = (tmp_path / "nested" / f"rank_{rank:04d}.log").read_text()
        assert "to file" in text

    def test_capture_warnings_routes_through_logger(self, isolated_config, capsys):
        kw = self._kw() | {"capture_warnings": True}
        log.configure_logging("INFO", colour=False, **kw)
        warnings.warn("careful now", UserWarning)
        assert "careful now" in capsys.readouterr().err

    def test_excepthook_logs_and_chains(self, isolated_config, capsys, monkeypatch):
        torn_down = []
        monkeypatch.setattr(log, "_teardown_process_group", lambda: torn_down.append(1))
        chained = []
        monkeypatch.setattr(sys, "excepthook", lambda *a: chained.append(a))
        kw = self._kw() | {"install_excepthook": True}
        log.configure_logging("INFO", colour=False, **kw)

        try:
            raise RuntimeError("kaput")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
        err = capsys.readouterr().err
        assert "uncaught exception" in err and "kaput" in err
        assert torn_down == [1] and chained == []

        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
        assert len(chained) == 1

    def test_faulthandler_is_enabled(self, isolated_config, monkeypatch):
        import faulthandler

        calls = []
        monkeypatch.setattr(faulthandler, "is_enabled", lambda: False)
        monkeypatch.setattr(faulthandler, "enable", lambda: calls.append("enable"))
        monkeypatch.setattr(faulthandler, "register", lambda sig: calls.append(sig))
        kw = self._kw() | {"enable_faulthandler": True}
        log.configure_logging("INFO", **kw)
        assert calls[0] == "enable" and len(calls) == 2


class TestHelpers:
    def test_timed_logs_duration(self, records, monkeypatch):
        monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
        with log.timed("phase"):
            pass
        (msg,) = _messages(records)
        assert msg.startswith("phase took ") and msg.endswith(" s")

    def test_timed_logs_even_if_body_raises(self, records, monkeypatch):
        monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
        with pytest.raises(ValueError):
            with log.timed("failing", level=logging.WARNING):
                raise ValueError
        assert records[0].levelno == logging.WARNING

    def test_timed_barriers_when_distributed(self, records, monkeypatch):
        calls = []
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "barrier", lambda group=None: calls.append(group))
        monkeypatch.setattr(log, "_rank_world", lambda: (0, 2))
        with log.timed("sync", barrier=True, cuda_sync=True, group="g"):
            pass
        assert calls == ["g", "g"]

    def test_die_logs_tears_down_and_exits(self, records, monkeypatch):
        exits, torn = [], []
        monkeypatch.setattr(log.os, "_exit", lambda code: exits.append(code))
        monkeypatch.setattr(log, "_teardown_process_group", lambda: torn.append(1))
        monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
        log.die("fatal %s", "x", code=3)
        assert exits == [3] and torn == [1]
        assert records[-1].levelno == logging.CRITICAL
        assert records[-1].getMessage() == "fatal x"

    def test_teardown_process_group_swallows_errors(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: True)

        def boom():
            raise RuntimeError("already gone")

        monkeypatch.setattr(dist, "destroy_process_group", boom)
        log._teardown_process_group()  # must not raise

    def test_teardown_process_group_is_noop_when_uninitialised(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        monkeypatch.setattr(
            dist, "destroy_process_group", lambda: pytest.fail("should not be called")
        )
        log._teardown_process_group()


class TestSingleProcessCollectives:
    """The collective helpers degrade gracefully when there is nothing to gather."""

    def test_log_ranks_logs_locally_without_a_group(self, records, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        monkeypatch.setattr(log, "_rank_world", lambda: (0, 1))
        log.log_ranks("hello %d", 1, level=logging.INFO)
        log.log_ranks("plain", level=logging.INFO)
        assert _messages(records) == ["hello 1", "plain"]

    def test_log_ranks_skips_disabled_levels(self, records, monkeypatch):
        monkeypatch.setattr(dist, "all_gather_object", lambda *a, **k: pytest.fail())
        log.get_logger().setLevel(logging.ERROR)
        log.log_ranks("nope", level=logging.DEBUG)
        assert records == []

    def test_assert_all_agree_outside_distributed(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        assert log.assert_all_agree("x", object()) is True

    def test_assert_all_agree_single_rank(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_world_size", lambda group=None: 1)
        assert log.assert_all_agree("x", 1) is True


@pytest.mark.dist(2)
class TestCollectivesTwoRanks:
    def test_log_ranks_gathers_in_rank_order(self, records):
        rank = dist.get_rank()
        log.log_ranks("value=%d", rank * 10, level=logging.INFO)
        dist.barrier()
        if rank == 0:
            (message,) = _messages(records)
            assert message == "rank dump:\n[0] value=0\n[1] value=10"
        else:
            assert records == []

    def test_assert_all_agree_true_when_equal(self):
        assert log.assert_all_agree("seed", 1234) is True

    def test_assert_all_agree_false_when_not_fatal(self, records):
        rank = dist.get_rank()
        assert log.assert_all_agree("seed", rank, fatal=False) is False
        dist.barrier()
        if rank == 0:
            assert "ranks disagree on seed: r0=0, r1=1" in _messages(records)[-1]

    def test_assert_all_agree_fatal_calls_die(self, monkeypatch):
        class DiedError(Exception):
            pass

        def fake_die(msg, *args, **kwargs):
            raise DiedError(msg % args)

        monkeypatch.setattr(log, "die", fake_die)
        with pytest.raises(DiedError, match="ranks disagree on 'seed'"):
            log.assert_all_agree("seed", dist.get_rank())
