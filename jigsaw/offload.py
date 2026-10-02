"""Activation offloading to host memory."""

import torch


class NVLinkActivationOffloader:
    """Offload activations to pinned CPU memory on a side stream.

    Device-to-host copies overlap with compute. Source tensors must stay alive
    until :meth:`wait_offloads` is called.
    """

    def __init__(self, device: torch.device) -> None:
        """Initialize the offloader.

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
        """Copy a tensor to pinned CPU memory asynchronously.

        Parameters
        ----------
        tensor : torch.Tensor
            GPU tensor, which must not be freed until :meth:`wait_offloads`.

        Returns
        -------
        torch.Tensor
            Pinned CPU tensor.
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

    def wait_offloads(self) -> None:
        """Block the current stream on pending copies and release GPU references."""
        cur = torch.cuda.current_stream(self.device)
        for _, ev in self._pending_d2h:
            cur.wait_event(ev)
        self._pending_d2h.clear()

    def prefetch(
        self, cpu_tensor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Copy a CPU tensor to the device asynchronously.

        Parameters
        ----------
        cpu_tensor : torch.Tensor
            Pinned CPU tensor.

        Returns
        -------
        gpu : torch.Tensor
            Tensor on the device.
        event : torch.cuda.Event
            Event recorded when the copy completes.
        """
        self.h2d_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.h2d_stream):
            gpu = cpu_tensor.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        return gpu, event

    def wait_prefetch(self, event: torch.cuda.Event) -> None:
        """Make the current stream wait for a prefetch to complete.

        Parameters
        ----------
        event : torch.cuda.Event
            Event returned by :meth:`prefetch`.
        """
        torch.cuda.current_stream(self.device).wait_event(event)
