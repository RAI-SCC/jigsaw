"""Tracing of the payloads that the distributed matmul kernels send.

Off by default: every hook returns after one ``None`` check until :func:`trace`
starts a tracer. Each send is labeled with the module that issued it (set by
:func:`scope`), the kernel (set by :func:`kernel`) and the payload kind passed at
the call site.
"""

import contextlib
import contextvars
import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import ParamSpec, TypeVar

import torch
import torch.distributed as dist

P = ParamSpec("P")
R = TypeVar("R")


@dataclasses.dataclass(frozen=True)
class CommRecord:
    """One payload sent to one peer.

    A tensor sent to several peers gives one record per peer.

    Attributes
    ----------
    step : int
        Value of :attr:`CommTracer.step` at the send.
    layer : str or None
        Name of the issuing module in the traced model, ``None`` outside a
        traced module.
    direction : str or None
        ``"forward"`` or ``"backward"``, ``None`` outside a traced module.
    kernel : str or None
        Name of the issuing kernel function.
    kind : str or None
        Operand slot of the payload in the kernel: ``"x"``, ``"w"``,
        ``"grad_output"`` or ``"partial"`` (a partial product).
    peer : int
        Destination rank within the process group.
    shape : tuple[int, ...]
        Shape of the payload.
    dtype : torch.dtype
        Data type of the payload.
    nbytes : int
        Bytes sent.
    """

    step: int
    layer: str | None
    direction: str | None
    kernel: str | None
    kind: str | None
    peer: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    nbytes: int


class CommTracer:
    """Collect a :class:`CommRecord` per send and pass each payload to ``hook``.

    Parameters
    ----------
    names : dict[int, str]
        Module names keyed by ``id(module)``.
    hook : Callable[[CommRecord, torch.Tensor], None], optional
        Called on every send with the record and the payload. The payload is
        the kernel's own tensor, which the kernel may reuse: clone it to keep it.
    """

    def __init__(
        self,
        names: dict[int, str],
        hook: Callable[[CommRecord, torch.Tensor], None] | None = None,
    ) -> None:
        """Initialize the tracer."""
        self.names = names
        self.hook = hook
        self.step = 0
        self.records: list[CommRecord] = []

    def next_step(self) -> None:
        """Advance the step label of later records."""
        self.step += 1

    def total_bytes(self) -> int:
        """Return the bytes sent by all records.

        Returns
        -------
        int
            Sum of :attr:`CommRecord.nbytes`.
        """
        return sum(r.nbytes for r in self.records)

    def _record(self, tensor: torch.Tensor, kind: str | None, peer: int) -> None:
        layer, direction = _scope.get() or (None, None)
        record = CommRecord(
            step=self.step,
            layer=layer,
            direction=direction,
            kernel=_kernel.get(),
            kind=kind,
            peer=peer,
            shape=tuple(tensor.shape),
            dtype=tensor.dtype,
            nbytes=tensor.numel() * tensor.element_size(),
        )
        self.records.append(record)
        if self.hook is not None:
            self.hook(record, tensor)


_tracer: CommTracer | None = None
#: ``(layer, direction)`` of the sends issued in the current block.
_scope: contextvars.ContextVar[tuple[str | None, str] | None] = contextvars.ContextVar(
    "_scope", default=None
)
#: Name of the kernel issuing the current sends.
_kernel: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_kernel", default=None
)


@contextlib.contextmanager
def trace(
    model: torch.nn.Module | None = None,
    hook: Callable[[CommRecord, torch.Tensor], None] | None = None,
) -> Iterator[CommTracer]:
    """Trace every labeled send issued inside the block.

    Parameters
    ----------
    model : torch.nn.Module, optional
        Model whose module names label the records.
    hook : Callable[[CommRecord, torch.Tensor], None], optional
        Called on every send; see :class:`CommTracer`.

    Yields
    ------
    CommTracer
        The active tracer.

    Raises
    ------
    RuntimeError
        If a trace is already active.
    """
    global _tracer
    if _tracer is not None:
        raise RuntimeError("a communication trace is already active")
    names = {} if model is None else {id(m): n for n, m in model.named_modules()}
    _tracer = CommTracer(names, hook)
    try:
        yield _tracer
    finally:
        _tracer = None


def layer_name(module: torch.nn.Module) -> str | None:
    """Return the name of ``module`` in the traced model.

    Parameters
    ----------
    module : torch.nn.Module
        Module issuing sends.

    Returns
    -------
    str or None
        Its name, or ``None`` without an active trace or outside the traced model.
    """
    return None if _tracer is None else _tracer.names.get(id(module))


def current_layer() -> str | None:
    """Return the layer label of the current block.

    Returns
    -------
    str or None
        The label set by the innermost :func:`scope`, or ``None``.
    """
    scope = _scope.get()
    return None if scope is None else scope[0]


def scope(layer: str | None, direction: str) -> contextlib.AbstractContextManager:
    """Label the sends issued inside the block with ``layer`` and ``direction``.

    Parameters
    ----------
    layer : str or None
        Layer label.
    direction : str
        ``"forward"`` or ``"backward"``.

    Returns
    -------
    contextlib.AbstractContextManager
        The labeling block; a no-op without an active trace.
    """
    if _tracer is None:
        return contextlib.nullcontext()
    return _set(_scope, (layer, direction))


def kernel(fn: Callable[P, R]) -> Callable[P, R]:
    """Label the sends issued inside ``fn`` with its name.

    Parameters
    ----------
    fn : Callable
        Communication kernel.

    Returns
    -------
    Callable
        ``fn``, setting the kernel label while a trace is active.
    """
    name = fn.__name__

    @functools.wraps(fn)
    def _wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if _tracer is None:
            return fn(*args, **kwargs)
        with _set(_kernel, name):
            return fn(*args, **kwargs)

    return _wrapper


def record_all_to_all(
    input_tensor_list: list[torch.Tensor],
    kind: str | None,
    group: dist.ProcessGroup | None,
) -> None:
    """Record the non-empty sends of an all-to-all to the other ranks.

    Parameters
    ----------
    input_tensor_list : list[torch.Tensor]
        Data sent to each rank of ``group``.
    kind : str or None
        Payload label.
    group : dist.ProcessGroup or None
        Process group of the exchange; the default group if ``None``.
    """
    if _tracer is None:
        return
    me = dist.get_rank() if group is None else group.rank()
    for peer, tensor in enumerate(input_tensor_list):
        if peer != me and tensor.numel() > 0:
            _tracer._record(tensor, kind, peer)


def record_send(
    tensor: torch.Tensor,
    kind: str | None,
    group: dist.ProcessGroup | None,
    group_dst: int | None = None,
    dst: int | None = None,
) -> None:
    """Record a point-to-point send to ``group_dst``, or to global rank ``dst``.

    Parameters
    ----------
    tensor : torch.Tensor
        Payload.
    kind : str or None
        Payload label.
    group : dist.ProcessGroup or None
        Process group of the send; the default group if ``None``.
    group_dst : int, optional
        Destination rank within ``group``.
    dst : int, optional
        Destination global rank, used if ``group_dst`` is ``None``.
    """
    if _tracer is None:
        return
    if group_dst is None:
        assert dst is not None
        group_dst = dst if group is None else dist.get_group_rank(group, dst)
    _tracer._record(tensor, kind, group_dst)


@contextlib.contextmanager
def _set(var: contextvars.ContextVar, value: object) -> Iterator[None]:
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)
