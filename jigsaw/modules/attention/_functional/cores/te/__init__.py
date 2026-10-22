"""TransformerEngine attention cores, picked by GPU architecture.

:mod:`.cudnn` (TE's fused kernels) on sm90+, :mod:`.flash` (``flash_attn``) on
sm80-89; see :func:`select_core`. Both implement the core contract in :mod:`..`.
Set ``JIGSAW_TE_CORE=cudnn|flash`` to override the choice.
"""

from __future__ import annotations

import os
from types import ModuleType

import torch

from . import cudnn, flash

#: Per-core availability: ``cudnn`` needs ``transformer_engine``, ``flash`` needs
#: ``flash_attn``.
CUDNN_CORE_AVAILABLE = cudnn.CUDNN_CORE_AVAILABLE
FLASH_CORE_AVAILABLE = flash.FLASH_CORE_AVAILABLE

#: Whether ``attn_backend="te"`` can run (either core importable).
TE_CORE_AVAILABLE = CUDNN_CORE_AVAILABLE or FLASH_CORE_AVAILABLE

_ENV_OVERRIDE = "JIGSAW_TE_CORE"
_CORES: dict[str, ModuleType] = {"cudnn": cudnn, "flash": flash}

#: First SM major version that gets the cuDNN core (9 = Hopper and newer).
_CUDNN_MIN_MAJOR = 9


def _available(core: ModuleType) -> bool:
    return bool(
        core.CUDNN_CORE_AVAILABLE  # type: ignore[attr-defined]
        if core is cudnn
        else core.FLASH_CORE_AVAILABLE  # type: ignore[attr-defined]
    )


def select_core(device: torch.device) -> ModuleType:
    """Pick the TE core for ``device``: :mod:`.cudnn` on sm90+, :mod:`.flash` below.

    Falls back to the other core if the preferred one is not importable. ROCm
    always gets :mod:`.flash`. ``JIGSAW_TE_CORE`` overrides the rule. Non-CUDA
    devices get :mod:`.cudnn`, whose kernel call then fails.

    Parameters
    ----------
    device : torch.device
        Device the attention runs on.

    Returns
    -------
    ModuleType
        The :mod:`.cudnn` or :mod:`.flash` core module.

    Raises
    ------
    ValueError
        If ``JIGSAW_TE_CORE`` names no core.
    RuntimeError
        If no suitable core is importable.
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
