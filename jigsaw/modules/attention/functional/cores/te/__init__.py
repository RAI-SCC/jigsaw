"""TransformerEngine attention cores, picked by GPU architecture.

``attn_backend="te"`` means "the kernel TransformerEngine would run here", and
that is not one kernel:

* :mod:`.cudnn` -- TE's own cuDNN fused kernels (``fused_attn_fwd`` /
  ``fused_attn_bwd``), on **Hopper and newer** (sm90+), where cuDNN ships
  architecture-specific FMHA kernels. Also the only TE path that can carry FP8.
* :mod:`.flash` -- the ``flash_attn`` package (Dao-AILab FA2), on **Ampere**
  (sm80/sm86/sm89). This mirrors what ``te.DotProductAttention`` does itself:
  its backend selector picks ``flash_attn`` over cuDNN for every non-FP8,
  no-mask, no-bias shape probed on an A100, and cuDNN was measurably slower
  there (see ``experiments/`` and the git history of this module).

:func:`select_core` is the whole rule. It is resolved once -- by
:func:`...config._select_attn_core_forward_fn` for the ring, and by
:meth:`~jigsaw.modules.attention.attention.DistributedAttention._select_attn_core`
for the non-ring core -- so nothing branches per call.

``attn_backend="sdpa"`` is deliberately *not* part of this: it means torch's own
kernels (:mod:`..sdpa` flash on CUDA, :mod:`..math` on CPU) on every
architecture, and stays the escape hatch when TE's choice is the wrong one for a
given shape.

Both cores expose the contract in :mod:`..` -- ``forward`` returns the softmax
LSE the ring's online-softmax merge needs, ``backward`` recomputes ``dQ/dK/dV``
from the saved *global* ``out``/``lse``, and dropout replay threads the kernel's
own opaque ``rng_state`` back rather than re-deriving a mask from a seed. Both
also take ``layout`` (``"bhsd"`` for the ring, ``"bshd"`` for the non-ring
core), since each underlying kernel speaks only ``bshd``.

Set ``JIGSAW_TE_CORE=cudnn`` or ``JIGSAW_TE_CORE=flash`` to override the
architecture rule -- mainly so both cores stay testable on whichever GPU is in
front of you, since the rule otherwise makes one of them unreachable.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch

from . import cudnn, flash

#: Per-core availability. ``cudnn`` needs ``transformer_engine``; ``flash`` needs
#: the ``flash_attn`` package. The container ships both.
CUDNN_CORE_AVAILABLE = cudnn.CUDNN_CORE_AVAILABLE
FLASH_CORE_AVAILABLE = flash.FLASH_CORE_AVAILABLE

#: Whether ``attn_backend="te"`` can run at all -- i.e. whether *either* arm is
#: importable. Read at call time by :func:`...config._te_core_or_raise` and at
#: import time by :mod:`jigsaw.modules.attention.attention`.
TE_CORE_AVAILABLE = CUDNN_CORE_AVAILABLE or FLASH_CORE_AVAILABLE

_ENV_OVERRIDE = "JIGSAW_TE_CORE"
_CORES: dict[str, ModuleType] = {"cudnn": cudnn, "flash": flash}

#: First Streaming-Multiprocessor major version that gets the cuDNN arm (9 =
#: Hopper; Blackwell and later fall here too, being newer than the Ampere
#: generation the flash arm was measured on).
_CUDNN_MIN_MAJOR = 9


def _available(core: ModuleType) -> bool:
    return bool(
        core.CUDNN_CORE_AVAILABLE  # type: ignore[attr-defined]
        if core is cudnn
        else core.FLASH_CORE_AVAILABLE  # type: ignore[attr-defined]
    )


def select_core(device: torch.device) -> ModuleType:
    """Pick the TE core for ``device``: :mod:`.cudnn` on sm90+, :mod:`.flash` below.

    Falls back to whichever arm is importable when the preferred one is not, so
    a container missing ``flash_attn`` still runs ``te`` on an A100 (on cuDNN,
    slower) rather than failing. ``JIGSAW_TE_CORE`` overrides the rule.

    Non-CUDA devices resolve to :mod:`.cudnn`, preserving the historical
    behaviour: the selector succeeds and the kernel call is what reports that
    there is no GPU.
    """
    override = os.environ.get(_ENV_OVERRIDE)
    if override is not None:
        if override not in _CORES:
            raise ValueError(
                f"{_ENV_OVERRIDE}={override!r}; expected one of {sorted(_CORES)}."
            )
        return _CORES[override]

    if device.type != "cuda":
        return cudnn

    preferred = (
        cudnn
        if torch.cuda.get_device_capability(device)[0] >= _CUDNN_MIN_MAJOR
        else flash
    )
    if _available(preferred):
        return preferred
    other = flash if preferred is cudnn else cudnn
    if _available(other):
        return other
    raise RuntimeError(
        "attn_backend=te needs transformer_engine (cuDNN core) or flash_attn "
        "(flash core); neither is importable."
    )
