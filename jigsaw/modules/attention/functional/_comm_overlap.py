"""Helpers for running the ring's point-to-point comm on a dedicated CUDA stream.

Gated by :attr:`~jigsaw.modules.attention.config.RingAttentionConfig.overlap_comm`
and used only by the ``SendOp.P2P`` / ``SendOp.ALLTOALL`` chunk providers (the
plain ring). The pattern mirrors :mod:`jigsaw.offload` and the stream pairs in
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
    produced there) is ready. Every tensor in ``keep_alive`` -- the send source
    and the receive buffer -- is ``record_stream``'d onto ``comm_stream`` so it
    survives until the transfer completes even after the Python reference goes
    away. ``event`` is recorded on ``comm_stream`` right after the comm is
    enqueued; the consumer does ``torch.cuda.current_stream().wait_event(event)``
    before touching ``recv_buf``.
    """
    compute = torch.cuda.current_stream()
    comm_stream.wait_stream(compute)
    with torch.cuda.stream(comm_stream):
        recv_buf, work = produce()
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
