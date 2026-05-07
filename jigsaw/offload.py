import torch


 
class NVLinkActivationOffloader:
    """
    Offloader with proper stream management.
 
    D2H copies run on a side stream so they overlap with compute.
    The source GPU tensor must stay alive until the copy finishes; we track
    pending copies via CUDA events and provide wait_offloads() to drain them
    before the caller drops references.
    """
 
    def __init__(self, device: torch.device):
        self.device = device
        self.d2h_stream = torch.cuda.Stream(device)
        self.h2d_stream = torch.cuda.Stream(device)
        # list of (gpu_tensor_ref, event) — gpu_tensor_ref keeps tensor alive
        # until the D2H copy finishes
        self._pending_d2h = []
 
    def offload(self, tensor: torch.Tensor) -> torch.Tensor:
        """
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
 
    def wait_offloads(self):
        """
        Block current stream on all pending D2H events, then drop GPU refs
        so the memory can actually be freed.
        """
        cur = torch.cuda.current_stream(self.device)
        for _, ev in self._pending_d2h:
            cur.wait_event(ev)
        self._pending_d2h.clear()
 
    def prefetch(self, cpu_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.cuda.Event]:
        """Async H2D on side stream. Returns (gpu_tensor, event)."""
        self.h2d_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.h2d_stream):
            gpu = cpu_tensor.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.h2d_stream)
        return gpu, event
 
    def wait_prefetch(self, event: torch.cuda.Event):
        """Make current stream wait on a prefetch to complete."""
        torch.cuda.current_stream(self.device).wait_event(event)
