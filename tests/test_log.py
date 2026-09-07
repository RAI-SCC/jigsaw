"""Unit tests for jigsaw.log rank-targeting (no distributed backend required)."""

import logging

import pytest

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
