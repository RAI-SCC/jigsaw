"""Helpers for running the ring's communication on a dedicated CUDA stream.

Gated by :attr:`~jigsaw.modules.RingAttentionConfig.overlap_comm`. Each
transfer runs on a side stream, its buffers are ``record_stream``'d so the
caching allocator does not recycle them mid-transfer, and the compute stream
waits on a CUDA event instead of the host blocking on ``Work.wait()``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist

#: Issues the async comm and returns ``(recv_buf, work)``, as
#: :func:`jigsaw._core.comm.comm_1d_p2p` does.
ProduceFn = Callable[[], "tuple[torch.Tensor, dist.Work | None]"]


def issue_on_comm_stream(
    comm_stream: torch.cuda.Stream,
    produce: ProduceFn,
    keep_alive: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dist.Work | None, torch.cuda.Event]:
    """Run ``produce`` on ``comm_stream`` and return ``(recv_buf, work, event)``.

    ``comm_stream`` first waits on the compute stream so the send source is
    ready. ``work.wait()`` then makes ``comm_stream``, not the host, wait for the
    transfer.

    Parameters
    ----------
    comm_stream : torch.cuda.Stream
        Side stream the transfer runs on.
    produce : ProduceFn
        Issues the comm op and returns ``(recv_buf, work)``.
    keep_alive : tuple[torch.Tensor, ...]
        Tensors the transfer reads, kept alive on ``comm_stream``.

    Returns
    -------
    recv_buf : torch.Tensor
        Buffer the transfer writes into.
    work : dist.Work or None
        Handle of the comm op, if it was asynchronous.
    event : torch.cuda.Event
        Recorded on ``comm_stream`` after the transfer; pass it to :func:`consume`.
    """
    compute = torch.cuda.current_stream()
    comm_stream.wait_stream(compute)
    with torch.cuda.stream(comm_stream):
        recv_buf, work = produce()
        if work is not None:
            work.wait()
    for t in keep_alive:
        t.record_stream(comm_stream)
    # Record after `work.wait()`: an earlier event would complete before the
    # transfer lands, letting the consumer read a buffer still being written.
    event = torch.cuda.Event()
    event.record(comm_stream)
    return recv_buf, work, event


def consume(event: torch.cuda.Event, *bufs: torch.Tensor) -> None:
    """Make the compute stream wait on ``event`` and adopt ``bufs``.

    Each buffer is ``record_stream``'d onto the compute stream, which reads it
    next.

    Parameters
    ----------
    event : torch.cuda.Event
        Event returned by :func:`issue_on_comm_stream`.
    *bufs : torch.Tensor
        Buffers the transfer wrote, about to be read on the compute stream.
    """
    compute = torch.cuda.current_stream()
    compute.wait_event(event)
    for b in bufs:
        b.record_stream(compute)


class PrefetchBuffer:
    """Double-buffered prefetch of one transfer at a time on the comm stream.

    The hybrid overlap providers use one per concurrent transfer (intra group
    and outer ring). :meth:`kick` starts a transfer into the buffer not
    returned by the last :meth:`wait`; :meth:`wait` returns the buffer of the
    last :meth:`kick`.

    Parameters
    ----------
    comm_stream : torch.cuda.Stream
        Side stream the transfers run on.
    like : torch.Tensor
        Template for the shape, dtype and device of both buffers.
    """

    def __init__(self, comm_stream: torch.cuda.Stream, like: torch.Tensor) -> None:
        """Allocate the two ping-pong buffers, shaped/dtype'd like ``like``."""
        self._cs = comm_stream
        self._bufs = [torch.empty_like(like), torch.empty_like(like)]
        self._pending = 0
        self._event: torch.cuda.Event | None = None
        self._work: object = None

    def kick(
        self,
        comm_call: Callable[[torch.Tensor], tuple[torch.Tensor, dist.Work | None]],
        extra_keep_alive: tuple[torch.Tensor, ...] = (),
    ) -> None:
        """Issue ``comm_call(buf)`` on the comm stream into the free buffer.

        Parameters
        ----------
        comm_call : Callable[[torch.Tensor], tuple[torch.Tensor, dist.Work or None]]
            Comm primitive taking the destination buffer.
        extra_keep_alive : tuple[torch.Tensor, ...], optional
            Source tensors ``comm_call`` reads besides the buffer.
        """
        into = 1 - self._pending
        buf = self._bufs[into]
        _, self._work, self._event = issue_on_comm_stream(
            self._cs, lambda: comm_call(buf), keep_alive=(buf, *extra_keep_alive)
        )
        self._pending = into

    def wait(self) -> torch.Tensor:
        """Make the compute stream wait for the pending transfer; return its buffer.

        Returns
        -------
        torch.Tensor
            Buffer the last :meth:`kick` wrote into.
        """
        assert self._event is not None
        buf = self._bufs[self._pending]
        consume(self._event, buf)
        return buf
