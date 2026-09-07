"""Process-wide cache of reusable communication buffers.

Distributed ops (matmul, convolution, ...) need scratch tensors to serve as
targets for asynchronous ``irecv`` / ``all_to_all`` calls. Re-allocating them
on every forward/backward pass is wasteful, so they are cached here and reused.

The cache is keyed on the *call site* (source file + line of the immediate
caller) plus ``device`` / ``dtype`` / ``local_rank`` / ``index`` -- deliberately
**not** on ``shape``. Every source line that requests a buffer corresponds to
one logical buffer slot; keying on shape/dtype/rank alone is unsafe because a
single call site routinely asks for several equally-shaped buffers that must
stay live simultaneously, and they would then alias and clobber each other.
``index`` disambiguates the rarer case of one source line producing several
buffers.

Caller contract: a buffer's contents must be fully consumed (e.g. copied out by
a matmul or ``torch.cat``) before the same call site requests a buffer again,
because the next request hands back the same storage.
"""

import sys

import torch
from torch._C._functorch import maybe_current_level

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
    """
    Retrieve (or lazily allocate) a reusable buffer for the calling site.

    A single flat allocation is cached and reused across calls so that repeated
    forward/backward passes do not keep re-allocating communication buffers. The
    buffer is keyed on the *call site* (the source file and line of the
    immediate caller) together with ``device`` / ``dtype`` / ``local_rank`` /
    ``index`` -- deliberately **not** on ``shape``.

    Keying on ``shape`` / ``dtype`` / ``local_rank`` alone is unsafe: nearly
    every call site asks for several buffers with an identical shape, dtype and
    rank that must stay live at the same time (e.g. concurrent ``irecv``
    targets). A key built only from those attributes hands the same storage out
    twice and the peers clobber each other's data. A call site, by contrast,
    corresponds to exactly one logical buffer, and ``index`` disambiguates the
    cases where one source line produces several buffers.

    Parameters
    ----------
    shape : torch.Size
        Shape of the view to return. The backing allocation is grown when the
        requested number of elements exceeds what is currently cached; a smaller
        request reuses the existing allocation and returns a trimmed view.
    dtype : torch.dtype
        Data type of the buffer.
    local_rank : int
        Local rank; keeps buffers belonging to different ranks that share a
        process from aliasing.
    device : torch.device
        Device the buffer lives on.
    index : int, optional
        Extra discriminator for multiple buffers requested from a single call
        site (default is 0).

    Returns
    -------
    torch.Tensor
        A zeroed view onto the cached buffer with the requested ``shape``.
    """
    needed = 1
    for dim in shape:
        needed *= dim

    # Under a functorch transform (``vmap`` / ``grad`` / ...) the module-level
    # cache tensor is "captured" -- the grad transform forbids the in-place
    # ``zero_`` below on it ("would mutate a captured Tensor"), and a cached
    # buffer would anyway be shared across the transform's batched slices. Hand
    # back a fresh zeroed tensor instead; the reuse optimisation only matters on
    # the plain eager path.
    if maybe_current_level() is not None:
        return torch.zeros(*shape, dtype=dtype, device=device)

    caller = sys._getframe(1)
    key: _BufferKey = (
        caller.f_code.co_filename,
        caller.f_lineno,
        str(device),
        dtype,
        local_rank,
        index,
    )

    buf = _prealloc_buffers.get(key)
    if buf is None or buf.numel() < needed:
        buf = torch.empty(needed, dtype=dtype, device=device)
        _prealloc_buffers[key] = buf

    out = buf[:needed]
    out.zero_()
    return out.view(*shape)


def clear_preallocated_buffers() -> None:
    """Drop every cached buffer, releasing the backing allocations."""
    _prealloc_buffers.clear()
