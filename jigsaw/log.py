"""Rank-aware logging.

Records carry the rank and elapsed time. Ordinary calls emit from one print
rank only, while :func:`at` and :func:`info_all` select other ranks.

Examples
--------
>>> from jigsaw import log
>>> log.configure_logging()  # doctest: +SKIP
>>> log.info("world_size=%d", 4)  # doctest: +SKIP
>>> log.at(2).debug("q shard %s", (2, 3))  # doctest: +SKIP
>>> log.info_all("process group ready")  # doctest: +SKIP

Notes
-----
:func:`configure_logging` reads these environment variables, which explicit
arguments override:

``JIGSAW_LOG_LEVEL``
    Level of the ``jigsaw`` logger (default ``"INFO"``).
``JIGSAW_LOG_RANK``
    Default print rank (default ``0``).
``JIGSAW_LOG_DIR``
    Directory for per-rank log files (default: none).
``JIGSAW_LOG_FORMAT``
    ``"json"`` for JSON lines, otherwise the human-readable format.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import platform
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, NoReturn

import torch
import torch.distributed as dist

__all__ = [
    "configure_logging",
    "get_logger",
    "set_print_rank",
    "at",
    "only_rank",
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "debug_all",
    "info_all",
    "warning_all",
    "error_all",
    "log_once",
    "log_ranks",
    "assert_all_agree",
    "timed",
    "die",
]

# ``"all"`` is a sentinel meaning "every rank"; otherwise a rank or ranks.
RankSpec = "int | Iterable[int] | str"

_LOGGER_NAME = "jigsaw"
_START = time.monotonic()

_configured = False
_print_rank = 0
_scope_ranks: Any = None  # RankSpec | None, set by only_rank()


# --------------------------------------------------------------------------- #
# rank resolution
# --------------------------------------------------------------------------- #
def _rank_world() -> tuple[int, int]:
    """Return ``(rank, world_size)`` from the process group or the launcher env."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    for rank_env, world_env in (
        (os.getenv("RANK"), os.getenv("WORLD_SIZE")),
        (os.getenv("SLURM_PROCID"), os.getenv("SLURM_NTASKS")),
    ):
        if rank_env is not None:
            return int(rank_env), int(world_env or 1)
    return 0, 1


def _local_rank() -> int:
    """Return the node-local rank from the launcher env (``0`` if unknown)."""
    return int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)


def _resolve_targets(spec: Any) -> set[int] | None:
    """Resolve a :data:`RankSpec` to a set of emitting ranks.

    ``None`` defers to :func:`only_rank`, then to the print rank. ``"all"``
    returns ``None``, meaning no rank filtering.
    """
    if spec is None:
        spec = _scope_ranks
    if spec is None:
        spec = _print_rank
    if isinstance(spec, str):
        return None
    if isinstance(spec, int):
        return {spec}
    return {int(r) for r in spec}


# --------------------------------------------------------------------------- #
# filter + formatters
# --------------------------------------------------------------------------- #
class _RankFilter(logging.Filter):
    """Stamp records with rank metadata and drop them on non-target ranks."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Annotate ``record`` and return whether the local rank may emit it."""
        rank, world = _rank_world()
        record.rank = rank  # type: ignore[attr-defined]
        record.world = world  # type: ignore[attr-defined]
        record.local_rank = _local_rank()  # type: ignore[attr-defined]
        record.elapsed = time.monotonic() - _START  # type: ignore[attr-defined]
        targets = _resolve_targets(getattr(record, "target_ranks", None))
        return targets is None or rank in targets


class _HumanFormatter(logging.Formatter):
    """Single-line human-readable formatter with optional color."""

    _COLOURS = {
        logging.DEBUG: "\033[38;5;245m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    _RESET = "\033[0m"

    def __init__(self, *, colour: bool) -> None:
        super().__init__()
        self._colour = colour

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a single line."""
        rec: Any = record
        name = record.name
        if name.startswith(_LOGGER_NAME + "."):
            name = name[len(_LOGGER_NAME) + 1 :]
        elif name == _LOGGER_NAME:
            name = "-"
        head = f"{rec.elapsed:7.2f}s [{rec.rank}/{rec.world}] {record.levelname:<7} {name}: "
        body = record.getMessage()
        if record.exc_info:
            body += "\n" + self.formatException(record.exc_info)
        line = head + body
        colour = self._COLOURS.get(record.levelno) if self._colour else None
        return f"{colour}{line}{self._RESET}" if colour else line


class _JsonFormatter(logging.Formatter):
    """Formatter emitting one JSON object per record."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a JSON object."""
        rec: Any = record
        payload: dict[str, Any] = {
            "t": round(rec.elapsed, 4),
            "rank": rec.rank,
            "world": rec.world,
            "level": record.levelname,
            "name": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def configure_logging(
    level: str | int | None = None,
    *,
    rank: int | None = None,
    log_dir: str | os.PathLike[str] | None = None,
    as_json: bool | None = None,
    colour: bool | None = None,
    capture_warnings: bool = True,
    install_excepthook: bool = True,
    enable_faulthandler: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Install the jigsaw logging handlers.

    Does nothing if already configured, unless ``force`` is set.

    Parameters
    ----------
    level : str | int, optional
        Level for the ``jigsaw`` logger. Falls back to ``$JIGSAW_LOG_LEVEL``
        then ``"INFO"``.
    rank : int, optional
        Print rank. Falls back to ``$JIGSAW_LOG_RANK`` then ``0``.
    log_dir : path-like, optional
        Directory for per-rank ``rank_<NNNN>.log`` files. Falls back to
        ``$JIGSAW_LOG_DIR``; no file is written if unset.
    as_json : bool, optional
        Emit JSON lines instead of the human format. Falls back to
        ``$JIGSAW_LOG_FORMAT == "json"``.
    colour : bool, optional
        Color the console stream. Defaults to on if stderr is a TTY and
        ``$NO_COLOR`` is unset.
    capture_warnings : bool
        Route :mod:`warnings` through logging.
    install_excepthook : bool
        Install a rank-aware ``sys.excepthook`` that logs the traceback and
        tears down the process group.
    enable_faulthandler : bool
        Enable :mod:`faulthandler` and, where supported, dump all thread stacks
        on ``SIGUSR1``.
    force : bool
        Reconfigure if already configured.

    Returns
    -------
    logging.Logger
        The ``jigsaw`` logger.
    """
    global _configured, _print_rank

    root = logging.getLogger(_LOGGER_NAME)
    if _configured and not force:
        return root

    if level is None:
        level = os.getenv("JIGSAW_LOG_LEVEL", "INFO")
    if rank is None:
        rank = int(os.getenv("JIGSAW_LOG_RANK", "0"))
    _print_rank = rank
    if log_dir is None:
        log_dir = os.getenv("JIGSAW_LOG_DIR") or None
    if as_json is None:
        as_json = os.getenv("JIGSAW_LOG_FORMAT", "").lower() == "json"
    if colour is None:
        colour = sys.stderr.isatty() and os.getenv("NO_COLOR") is None

    rank_filter = _RankFilter()

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        _JsonFormatter() if as_json else _HumanFormatter(colour=colour)
    )
    console.addFilter(rank_filter)
    handlers: list[logging.Handler] = [console]

    if log_dir is not None:
        my_rank, _ = _rank_world()
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            directory / f"rank_{my_rank:04d}.log", mode="w"
        )
        file_handler.setFormatter(
            _JsonFormatter() if as_json else _HumanFormatter(colour=False)
        )
        file_handler.addFilter(rank_filter)
        handlers.append(file_handler)

    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False

    # Third-party chatter belongs one level up.
    logging.getLogger("torch").setLevel(logging.WARNING)

    if capture_warnings:
        logging.captureWarnings(True)
        py_warnings = logging.getLogger("py.warnings")
        for handler in handlers:
            py_warnings.addHandler(handler)

    if enable_faulthandler and not faulthandler.is_enabled():
        faulthandler.enable()
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if sigusr1 is not None:
            faulthandler.register(sigusr1)

    if install_excepthook:
        _install_excepthook()

    _configured = True
    root.debug(
        "logging configured (level=%s, print_rank=%d, log_dir=%s, json=%s)",
        level,
        _print_rank,
        log_dir,
        as_json,
    )
    return root


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child of the ``jigsaw`` logger.

    Parameters
    ----------
    name : str, optional
        Sub-name such as ``"attention.ring"``. ``None`` returns the ``jigsaw``
        logger.

    Returns
    -------
    logging.Logger
        Logger.
    """
    if not name:
        return logging.getLogger(_LOGGER_NAME)
    if name == _LOGGER_NAME or name.startswith(_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")


_base = logging.getLogger(_LOGGER_NAME)


# --------------------------------------------------------------------------- #
# rank-targeted logging
# --------------------------------------------------------------------------- #
class _RankLogger:
    """Logging proxy bound to a :data:`RankSpec`."""

    __slots__ = ("_ranks",)

    def __init__(self, ranks: Any) -> None:
        self._ranks = ranks

    def _log(
        self,
        level: int,
        msg: str,
        args: tuple[object, ...],
        *,
        exc_info: bool = False,
        stacklevel: int = 2,
    ) -> None:
        if not _base.isEnabledFor(level):
            return
        _base.log(
            level,
            msg,
            *args,
            exc_info=exc_info,
            extra={"target_ranks": self._ranks},
            stacklevel=stacklevel + 1,
        )

    def debug(self, msg: str, *args: object) -> None:
        """Log ``msg % args`` at DEBUG from this proxy's ranks."""
        self._log(logging.DEBUG, msg, args)

    def info(self, msg: str, *args: object) -> None:
        """Log ``msg % args`` at INFO from this proxy's ranks."""
        self._log(logging.INFO, msg, args)

    def warning(self, msg: str, *args: object) -> None:
        """Log ``msg % args`` at WARNING from this proxy's ranks."""
        self._log(logging.WARNING, msg, args)

    def error(self, msg: str, *args: object) -> None:
        """Log ``msg % args`` at ERROR from this proxy's ranks."""
        self._log(logging.ERROR, msg, args)

    def exception(self, msg: str, *args: object) -> None:
        """Log ``msg % args`` at ERROR with the active exception's traceback."""
        self._log(logging.ERROR, msg, args, exc_info=True)


_DEFAULT = _RankLogger(None)
_ALL = _RankLogger("all")

debug = _DEFAULT.debug
info = _DEFAULT.info
warning = _DEFAULT.warning
error = _DEFAULT.error
exception = _DEFAULT.exception

debug_all = _ALL.debug
info_all = _ALL.info
warning_all = _ALL.warning
error_all = _ALL.error


def at(ranks: Any) -> _RankLogger:
    """Return a logging proxy that emits from ``ranks`` only.

    Parameters
    ----------
    ranks : int | Iterable[int] | "all"
        Ranks that emit records.

    Returns
    -------
    _RankLogger
        Logging proxy.

    Examples
    --------
    >>> at(2).info("only rank 2")  # doctest: +SKIP
    >>> at([0, 3]).warning("ranks 0 and 3")  # doctest: +SKIP
    """
    return _RankLogger(ranks)


def set_print_rank(rank: int) -> None:
    """Set the print rank.

    Parameters
    ----------
    rank : int
        Rank that ordinary log calls emit from.
    """
    global _print_rank
    _print_rank = rank


@contextmanager
def only_rank(ranks: Any) -> Iterator[None]:
    """Emit default log calls from ``ranks`` inside the block.

    Parameters
    ----------
    ranks : int | Iterable[int] | "all"
        Ranks that emit inside the block.
    """
    global _scope_ranks
    previous = _scope_ranks
    _scope_ranks = ranks
    try:
        yield
    finally:
        _scope_ranks = previous


_seen_once: set[tuple[str, int, str]] = set()


def log_once(msg: str, *args: object, level: int = logging.INFO) -> None:
    """Log ``msg`` only the first time a call site runs.

    Parameters
    ----------
    msg : str
        ``%``-style format string.
    *args : object
        Format arguments.
    level : int
        Logging level.
    """
    frame: FrameType | None = sys._getframe(1)
    filename = frame.f_code.co_filename if frame else "?"
    lineno = frame.f_lineno if frame else 0
    key = (filename, lineno, msg)
    if key in _seen_once:
        return
    _seen_once.add(key)
    _DEFAULT._log(level, msg, args, stacklevel=1)


def log_ranks(
    msg: str,
    *args: object,
    level: int = logging.DEBUG,
    group: dist.ProcessGroup | None = None,
) -> None:
    """Gather one message per rank and log them in rank order from rank 0.

    Collective: must be called on every rank in ``group``.

    Parameters
    ----------
    msg : str
        ``%``-style format string, formatted independently on each rank.
    *args : object
        Format arguments.
    level : int
        Logging level. Nothing is gathered if the level is disabled.
    group : torch.distributed.ProcessGroup, optional
        Process group. Defaults to the world group.
    """
    if not _base.isEnabledFor(level):
        return
    rank, world = _rank_world()
    line = (msg % args) if args else msg
    if not (dist.is_available() and dist.is_initialized()) or world == 1:
        _DEFAULT._log(level, "%s", (line,), stacklevel=1)
        return
    gathered: list[Any] = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, f"[{rank}] {line}", group=group)
    if dist.get_rank(group) == 0:
        body = "\n".join(str(item) for item in gathered)
        _ALL._log(level, "rank dump:\n%s", (body,), stacklevel=1)


def assert_all_agree(
    name: str,
    value: object,
    *,
    group: dist.ProcessGroup | None = None,
    fatal: bool = True,
) -> bool:
    """Check that ``value`` is identical on every rank.

    Collective: must be called on every rank in ``group``.

    Parameters
    ----------
    name : str
        Label used in the log message.
    value : object
        Rank-local value. Must be picklable and support ``==``.
    group : torch.distributed.ProcessGroup, optional
        Process group. Defaults to the world group.
    fatal : bool
        Call :func:`die` on mismatch instead of logging an error and returning
        ``False``.

    Returns
    -------
    bool
        Whether all ranks agree. Always ``True`` outside distributed runs.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return True
    world = dist.get_world_size(group)
    if world == 1:
        return True
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, value, group=group)
    if all(item == gathered[0] for item in gathered):
        return True
    if dist.get_rank(group) == 0:
        detail = ", ".join(f"r{i}={v!r}" for i, v in enumerate(gathered))
        _ALL._log(
            logging.ERROR, "ranks disagree on %s: %s", (name, detail), stacklevel=1
        )
    if fatal:
        die("ranks disagree on %r", name)
    return False


@contextmanager
def timed(
    label: str,
    *,
    level: int = logging.INFO,
    barrier: bool = False,
    cuda_sync: bool = False,
    group: dist.ProcessGroup | None = None,
) -> Iterator[None]:
    """Time a block and log its duration from the print rank.

    Parameters
    ----------
    label : str
        Name of the timed region.
    level : int
        Level of the result line.
    barrier : bool
        Call ``dist.barrier`` before and after the block.
    cuda_sync : bool
        Call ``torch.cuda.synchronize`` before stopping the clock.
    group : torch.distributed.ProcessGroup, optional
        Process group for the barriers.
    """
    in_dist = dist.is_available() and dist.is_initialized()
    if barrier and in_dist:
        dist.barrier(group=group)
    start = time.perf_counter()
    try:
        yield
    finally:
        if cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        if barrier and in_dist:
            dist.barrier(group=group)
        _DEFAULT._log(
            level, "%s took %.3f s", (label, time.perf_counter() - start), stacklevel=1
        )


def die(msg: str, *args: object, code: int = 1) -> NoReturn:
    """Log a fatal error, tear down the process group and exit.

    Exits with ``os._exit`` after flushing the handlers.

    Parameters
    ----------
    msg : str
        ``%``-style format string.
    *args : object
        Format arguments.
    code : int
        Exit status.
    """
    _ALL._log(logging.CRITICAL, msg, args, stacklevel=1)
    for handler in logging.getLogger(_LOGGER_NAME).handlers:
        handler.flush()
    _teardown_process_group()
    os._exit(code)


# --------------------------------------------------------------------------- #
# crash / hang diagnosis
# --------------------------------------------------------------------------- #
def _teardown_process_group() -> None:
    """Destroy the process group, ignoring errors."""
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        dist.destroy_process_group()
    except Exception:  # noqa: BLE001 - shutdown path, nothing useful to do
        pass


def _install_excepthook() -> None:
    """Install a rank-aware ``sys.excepthook`` chained to the previous one."""
    previous = sys.excepthook

    def hook(
        exc_type: type[BaseException],
        exc: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            previous(exc_type, exc, traceback)
            return
        rank, world = _rank_world()
        logging.getLogger(_LOGGER_NAME).critical(
            "uncaught exception on rank %d/%d (%s)",
            rank,
            world,
            platform.node(),
            exc_info=(exc_type, exc, traceback),
            extra={"target_ranks": "all"},
        )
        for handler in logging.getLogger(_LOGGER_NAME).handlers:
            handler.flush()
        _teardown_process_group()

    sys.excepthook = hook
