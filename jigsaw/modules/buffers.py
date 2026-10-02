"""Cache of reusable communication buffers.

Buffers are keyed on the call site (file and line of the caller) plus device,
dtype, local rank and index, but not on shape. A buffer must be consumed before
the same call site requests it again, since the storage is reused.
"""

import sys

import torch

# (caller_file, caller_lineno, str(device), dtype, local_rank, index)
_BufferKey = tuple[str, int, str, torch.dtype, int, int]
_prealloc_buffers: dict[_BufferKey, torch.Tensor] = {}


def get_preallocated_buffer(
    shape: torch.Size,
    dtype: torch.dtype,
    local_rank: int,
    device: torch.device,
    index: int = 0,
) -> torch.Tensor:
    """Return a reusable buffer for the calling site.

    The backing allocation is cached and grown when a larger buffer is
    requested.

    Parameters
    ----------
    shape : torch.Size
        Shape of the returned view.
    dtype : torch.dtype
        Data type of the buffer.
    local_rank : int
        Local rank, so that ranks sharing a process do not share buffers.
    device : torch.device
        Device the buffer lives on.
    index : int
        Discriminator for several buffers requested from one call site.

    Returns
    -------
    torch.Tensor
        Zeroed view of the cached buffer.
    """
    caller = sys._getframe(1)
    key: _BufferKey = (
        caller.f_code.co_filename,
        caller.f_lineno,
        str(device),
        dtype,
        local_rank,
        index,
    )

    needed = 1
    for dim in shape:
        needed *= dim

    buf = _prealloc_buffers.get(key)
    if buf is None or buf.numel() < needed:
        buf = torch.empty(needed, dtype=dtype, device=device)
        _prealloc_buffers[key] = buf

    out = buf[:needed]
    out.zero_()
    return out.view(*shape)


def clear_preallocated_buffers() -> None:
    """Release all cached buffers."""
    _prealloc_buffers.clear()
