"""TransformerEngine attention cores, picked by GPU architecture.

``attn_backend="te"`` is not one kernel: :mod:`.cudnn` (TE's fused
``fused_attn_fwd``/``bwd``, Hopper+/sm90+, the only TE path with FP8) or
:mod:`.flash` (``flash_attn`` FA2, Ampere/sm80-89 -- mirrors
``te.DotProductAttention``'s own choice there, where cuDNN measured slower).
:func:`select_core` is the whole rule, resolved once per module (ring:
``..._config._select_attn_core_forward_fn``; non-ring:
:meth:`~jigsaw.modules.DistributedAttention._select_attn_core`).

``attn_backend="sdpa"`` is separate: torch's own kernels on every architecture,
the escape hatch when TE's pick is wrong for a given shape.

Both cores implement the contract in :mod:`..` (LSE for the ring's
online-softmax merge, backward recompute from saved global ``out``/``lse``,
opaque kernel ``rng_state`` for dropout replay, and a ``layout`` arg since each
kernel only speaks ``bshd``).

Override the architecture rule with ``JIGSAW_TE_CORE=cudnn|flash`` to keep
both cores testable on one GPU.
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
#: importable. Read at call time by :func:`..._config._te_core_or_raise` and at
#: import time by :mod:`jigsaw.modules.attention._attention`.
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

    # On ROCm the capability is the gfx version (MI250/MI300 report 9), so the
    # Hopper rule does not apply, and the cuDNN arm's TE calls are unverified there.
    if torch.version.hip is not None:
        if _available(flash):
            return flash
        raise RuntimeError(
            "attn_backend=te on ROCm needs flash_attn; use attn_backend='sdpa', "
            f"or set {_ENV_OVERRIDE}=cudnn to try TE's fused core (untested)."
        )

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
