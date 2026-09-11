"""Helpers for running the ring's point-to-point comm on a dedicated CUDA stream.

Gated by :attr:`~jigsaw.modules.attention.config.RingAttentionConfig.overlap_comm`.
Originally built for (and still used directly by) the plain ``SendOp.P2P`` /
``SendOp.ALLTOALL`` chunk providers; :class:`PrefetchBuffer` below generalizes
the same ping-pong pattern for the hybrid (intra-/inter-node) providers, which
have more than one concurrent hop in flight. The pattern mirrors
:mod:`jigsaw.offload` and the stream pairs in
:mod:`jigsaw.modules.linear.distributed_matmul`: issue the transfer on a side
stream, tag the buffers with ``record_stream`` so the caching allocator will not
recycle them mid-flight, and hand back a CUDA event the consumer waits on from
the compute stream instead of blocking the host on ``Work.wait()``.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.distributed as dist

#: ``produce()`` issues the async comm and returns ``(recv_buf, work)`` -- the
#: exact shape :func:`jigsaw.comm.comm_1d_p2p` / :func:`~jigsaw.comm.comm_1d_alltoall`
#: already return.
ProduceFn = Callable[[], "tuple[torch.Tensor, Optional[dist.Work]]"]


def issue_on_comm_stream(
    comm_stream: torch.cuda.Stream,
    produce: ProduceFn,
    keep_alive: "tuple[torch.Tensor, ...]",
) -> "tuple[torch.Tensor, Optional[dist.Work], torch.cuda.Event]":
    """Run ``produce`` on ``comm_stream`` and return ``(recv_buf, work, event)``.

    ``comm_stream`` first waits on the compute stream so the send source (freshly
    produced there) is ready. ``produce`` issues the actual NCCL op, which
    ProcessGroupNCCL always runs on its *own* internal stream, not whichever one
    is "current" when the call is made -- so the ``torch.cuda.stream(comm_stream)``
    context here only decides what that internal stream is told to wait for
    beforehand, not where the transfer itself completes. ``work.wait()`` is what
    bridges the two: called with ``comm_stream`` current, it inserts a wait for
    NCCL's real completion event onto ``comm_stream`` (a stream-side wait, not a
    host block -- ``Work.wait()`` only blocks the calling CPU thread when there is
    no current-stream event to enqueue against). *Only after that* is it safe to
    record ``event`` on ``comm_stream``: recording it any earlier (e.g. right
    after ``produce()`` returns, before the transfer it kicked off has actually
    been waited onto this stream) captures a point ``comm_stream`` reaches
    immediately, before the real transfer lands -- the consumer would then adopt
    the buffer while it is still being written. Every tensor in ``keep_alive`` --
    the send source and the receive buffer -- is ``record_stream``'d onto
    ``comm_stream`` so it survives until the transfer completes even after the
    Python reference goes away. The consumer does
    ``torch.cuda.current_stream().wait_event(event)`` before touching ``recv_buf``
    (see :func:`consume`).
    """
    compute = torch.cuda.current_stream()
    comm_stream.wait_stream(compute)
    with torch.cuda.stream(comm_stream):
        recv_buf, work = produce()
        if work is not None:
            work.wait()
    for t in keep_alive:
        t.record_stream(comm_stream)
    event = torch.cuda.Event()
    event.record(comm_stream)
    return recv_buf, work, event


def consume(event: torch.cuda.Event, *bufs: torch.Tensor) -> None:
    """Make the compute stream wait on ``event`` and adopt ``bufs``.

    The mirror of :func:`issue_on_comm_stream`: the compute stream blocks until
    the recorded transfer is done, then each buffer is ``record_stream``'d back
    onto the compute stream (it is about to be read there).
    """
    compute = torch.cuda.current_stream()
    compute.wait_event(event)
    for b in bufs:
        b.record_stream(compute)


class PrefetchBuffer:
    """Double-buffered, one-hop-ahead comm-stream prefetch.

    Generalizes the ping-pong buffering the plain-ring overlap providers
    (:class:`~.forward_chunk_provider._RingForwardChunkProviderOverlap`,
    :class:`~.backward_chunk_provider._LocalChunkProviderOverlap`) hand-roll, so
    every hybrid-ring overlap variant -- which juggles more than one concurrent
    hop (an intra-node ring/broadcast *and* an inter-node ring) -- gets one
    instance per hop instead of duplicating the buffer/event bookkeeping. Two
    persistent buffers shaped like ``like`` ping-pong: :meth:`kick` issues a
    transfer into whichever one is currently free, :meth:`wait` adopts the
    other (the one the previous :meth:`kick` targeted).
    """

    def __init__(self, comm_stream: torch.cuda.Stream, like: torch.Tensor) -> None:
        """Allocate the two ping-pong buffers, shaped/dtype'd like ``like``."""
        self._cs = comm_stream
        self._bufs = [torch.empty_like(like), torch.empty_like(like)]
        self._pending = 0
        self._event: Optional[torch.cuda.Event] = None
        self._work: object = None

    def kick(
        self,
        comm_call: "Callable[[torch.Tensor], tuple[torch.Tensor, Optional[dist.Work]]]",
        extra_keep_alive: "tuple[torch.Tensor, ...]" = (),
    ) -> None:
        """Issue ``comm_call(buf)`` on the comm stream into the free buffer.

        ``comm_call`` is the concrete comm primitive bound to everything except
        its destination buffer -- e.g. ``lambda buf: config.send_op_fn(src, buf,
        direction=-1, group=group, async_op=True, tag=tag)``. ``extra_keep_alive``
        names any source tensor ``comm_call`` reads besides ``buf`` itself (e.g.
        ``src`` above), so it is ``record_stream``'d too -- see
        :func:`issue_on_comm_stream`.
        """
        into = 1 - self._pending
        buf = self._bufs[into]
        _, self._work, self._event = issue_on_comm_stream(
            self._cs, lambda: comm_call(buf), keep_alive=(buf, *extra_keep_alive)
        )
        self._pending = into

    def wait(self) -> torch.Tensor:
        """Make the compute stream wait for the pending transfer; return its buffer."""
        assert self._event is not None
        buf = self._bufs[self._pending]
        consume(self._event, buf)
        return buf
