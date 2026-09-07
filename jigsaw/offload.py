"""Offloading class."""

import torch


class NVLinkActivationOffloader:
    """
    Offloader with proper stream management.

    D2H copies run on a side stream so they overlap with compute.
    The source GPU tensor must stay alive until the copy finishes; we track
    pending copies via CUDA events and provide wait_offloads() to drain them
    before the caller drops references.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize the object.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.

        """
        self.device = device
        self.d2h_stream = torch.cuda.Stream(device)
        self.h2d_stream = torch.cuda.Stream(device)
        # list of (gpu_tensor_ref, event) — gpu_tensor_ref keeps tensor alive
        # until the D2H copy finishes
        self._pending_d2h: list[tuple[torch.Tensor, torch.cuda.Event]] = []
        self.name = "NVLinkActivationOffloader"

    def offload(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Offload tensor to CPU.

        Async D2H copy on side stream. Returns pinned CPU tensor.
        IMPORTANT: caller must NOT free `tensor` until wait_offloads() is called,
        because the copy is still in flight on the side stream.
        """
        cpu = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu"
        )
        # ensure D2H stream waits for the compute that produced `tensor`
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            cpu.copy_(tensor, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.d2h_stream)
        # hold a ref to keep tensor alive across the async copy
        self._pending_d2h.append((tensor, event))
        return cpu

    def offload_blocking(self, tensor: torch.Tensor) -> torch.Tensor:
        """D2H-copy ``tensor`` into pinned host memory, returning once it lands.

        Unlike :meth:`offload` this is synchronous: the copy runs on the D2H
        side stream (so it still waits for the compute that produced ``tensor``)
        but the CPU blocks until it completes, so the returned pinned tensor is
        immediately safe to hand to a host-side collective (e.g. a Gloo
        ``isend``) and ``tensor`` may be freed right away -- nothing is left in
        ``_pending_d2h``.
        """
        cpu = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=True, device="cpu"
        )
        self.d2h_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.d2h_stream):
            cpu.copy_(tensor, non_blocking=True)
        self.d2h_stream.synchronize()
        return cpu

    def prefetch_into(
        self, dst_gpu: torch.Tensor, src_cpu: torch.Tensor
    ) -> torch.cuda.Event:
        """Async H2D-copy ``src_cpu`` into the existing GPU buffer ``dst_gpu``.

        Like :meth:`prefetch` but writes into a caller-provided buffer instead
        of allocating a fresh one. Returns an event on the H2D stream; pass it
        to :meth:`wait_prefetch` before reading ``dst_gpu``.
        """
        self.h2d_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.h2d_stream):
            dst_gpu.copy_(src_cpu, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        return event

    def wait_offloads(self) -> None:
        """
        Wait to offload.

        Block current stream on all pending D2H events, then drop GPU refs
        so the memory can actually be freed.
        """
        cur = torch.cuda.current_stream(self.device)
        for _, ev in self._pending_d2h:
            cur.wait_event(ev)
        self._pending_d2h.clear()

    def prefetch(
        self, cpu_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Prefetch tensor from CPU.

        Async H2D on side stream. Returns (gpu_tensor, event).
        """
        self.h2d_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.h2d_stream):
            gpu = cpu_tensor.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        return gpu, event

    def wait_prefetch(self, event: torch.cuda.Event) -> None:
        """Wait for prefetching.

        Make current stream wait on a prefetch to complete.
        """
        torch.cuda.current_stream(self.device).wait_event(event)
