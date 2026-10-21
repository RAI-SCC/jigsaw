"""Pin jigsaw's public API so the internal layout can be refactored safely.

The public surface is the ``__all__`` of the modules in :data:`EXPECTED_ALL`;
everything below ``jigsaw.modules.<sub>`` lives in ``_``-prefixed internal
modules that may move. Every public name carries the ``__module__`` of the path
it is documented under (see :func:`jigsaw._core.public.set_module`), so reprs
and pickles never name an internal file.

A failure here means the public API changed. If that is intended, update
:data:`EXPECTED_ALL`; it is a breaking change for users when a name is removed
or a label moves.
"""

import importlib
import inspect
import pkgutil

import pytest
import torch

import jigsaw
import jigsaw.modules

SUBPACKAGES = (
    "activation",
    "attention",
    "convolution",
    "linear",
    "loss",
    "metrics",
    "normalization",
    "permute",
    "roll",
)

EXPECTED_ALL: dict[str, set[str]] = {
    "jigsaw": {
        "ProcessMesh",
        "make_process_mesh",
        "convert",
        "debug",
        "gather",
        "init",
        "launch",
        "log",
        "modules",
        "parallel",
    },
    "jigsaw.parallel": {
        "JointCommHookState",
        "broadcast_replicas",
        "check_ddp",
        "joint_comm_hook",
        "mark_replicated",
        "register_joint_comm_hook",
        "parameter_count",
        "replica_groups",
        "sync_grad",
        "synced_call",
        "synced_parameters",
        "wrap_ddp",
    },
    "jigsaw.modules": {
        # Layers
        "Attention",
        "DistributedAttention",
        "DistributedConv1d",
        "DistributedConv2d",
        "DistributedConv3d",
        "DistributedConvTranspose1d",
        "DistributedConvTranspose2d",
        "DistributedConvTranspose3d",
        "DistributedGroupNorm",
        "DistributedLayerNorm",
        "DistributedLinear",
        "DistributedMatMulXTW",
        "DistributedMatMulXW",
        "DistributedMatMulXWT",
        "DistributedRoll",
        "DistributedSoftmax",
        "MatMulXTW",
        "MatMulXW",
        "MatMulXWT",
        "Roll",
        # Losses
        "DistributedBCELoss",
        "DistributedBCEWithLogitsLoss",
        "DistributedCrossEntropyLoss",
        "DistributedL1Loss",
        "DistributedMSELoss",
        "DistributedNLLLoss",
        "DistributedSmoothL1Loss",
        # Metrics
        "DistributedSegmentationMetrics",
        # Configs
        "ConvNdConfig",
        "ConvTransposeNdConfig",
        "LinearConfig",
        "RingAttentionConfig",
        "RollConfig",
        # Subpackages
        "functional",
        *SUBPACKAGES,
    },
    "jigsaw.modules.functional": {
        "confusion_to_metrics",
        "distributed_argmax",
        "distributed_binary_cross_entropy",
        "distributed_binary_cross_entropy_with_logits",
        "distributed_confusion_matrix",
        "distributed_cross_entropy",
        "distributed_l1_loss",
        "distributed_log_softmax",
        "distributed_mse_loss",
        "distributed_nll_loss",
        "distributed_permute",
        "distributed_smooth_l1_loss",
        "redundant_all_reduce",
    },
    "jigsaw.modules.activation": {
        "DistributedSoftmax",
        "distributed_argmax",
        "distributed_log_softmax",
    },
    "jigsaw.modules.attention": {
        "Attention",
        "AttnBackend",
        "DistributedAttention",
        "JIGSAW_TE_AVAILABLE",
        "Offload",
        "RingAttentionConfig",
        "SaveBackward",
        "SaveScope",
        "SendBackward",
        "SendForward",
        "SendOp",
    },
    "jigsaw.modules.convolution": {
        "ConvNdConfig",
        "ConvTransposeNdConfig",
        "DistributedConv1d",
        "DistributedConv2d",
        "DistributedConv3d",
        "DistributedConvTranspose1d",
        "DistributedConvTranspose2d",
        "DistributedConvTranspose3d",
    },
    "jigsaw.modules.linear": {
        "DistributedLinear",
        "DistributedMatMulXTW",
        "DistributedMatMulXW",
        "DistributedMatMulXWT",
        "LinearConfig",
        "MatMulXTW",
        "MatMulXW",
        "MatMulXWT",
    },
    "jigsaw.modules.loss": {
        "DistributedBCELoss",
        "DistributedBCEWithLogitsLoss",
        "DistributedCrossEntropyLoss",
        "DistributedL1Loss",
        "DistributedMSELoss",
        "DistributedNLLLoss",
        "DistributedSmoothL1Loss",
        "distributed_binary_cross_entropy",
        "distributed_binary_cross_entropy_with_logits",
        "distributed_cross_entropy",
        "distributed_l1_loss",
        "distributed_mse_loss",
        "distributed_nll_loss",
        "distributed_smooth_l1_loss",
        "redundant_all_reduce",
    },
    "jigsaw.modules.metrics": {
        "DistributedSegmentationMetrics",
        "confusion_to_metrics",
        "distributed_confusion_matrix",
    },
    "jigsaw.modules.normalization": {
        "DistributedGroupNorm",
        "DistributedLayerNorm",
    },
    "jigsaw.modules.permute": {"distributed_permute"},
    "jigsaw.modules.roll": {"DistributedRoll", "Roll", "RollConfig"},
}

PUBLIC_MODULES = sorted(EXPECTED_ALL)


def _public_objects() -> list[tuple[str, str, object]]:
    """Every (module, name, object) exported by a public module, submodules excluded."""
    out = []
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            obj = getattr(module, name)
            if not inspect.ismodule(obj):
                out.append((module_name, name, obj))
    return out


def _canonical_module(module_name: str, name: str) -> str:
    """The public path a name is documented, and labelled, under."""
    if name in EXPECTED_ALL["jigsaw.modules"]:
        return "jigsaw.modules"
    if name in EXPECTED_ALL["jigsaw.modules.functional"]:
        return "jigsaw.modules.functional"
    return module_name


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_all_is_pinned(module_name: str) -> None:
    """``__all__`` matches the pinned surface and has no duplicates."""
    module = importlib.import_module(module_name)
    assert len(module.__all__) == len(set(module.__all__)), "duplicate in __all__"
    assert set(module.__all__) == EXPECTED_ALL[module_name]


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_all_names_resolve(module_name: str) -> None:
    """``from <module> import <name>`` works for every exported name."""
    module = importlib.import_module(module_name)
    missing = [name for name in module.__all__ if not hasattr(module, name)]
    assert not missing


@pytest.mark.parametrize(
    "module_name, name, obj",
    [
        (m, n, o)
        for m, n, o in _public_objects()
        if getattr(o, "__module__", "").startswith("jigsaw.")
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_label_is_public_and_resolves(module_name: str, name: str, obj: object) -> None:
    """``__module__`` names the public path, and importing it finds the same object.

    That is what pickles and reprs use, so it must never name an internal file.
    """
    expected = _canonical_module(module_name, name)
    if module_name == "jigsaw":
        # Utility modules define their names in place, e.g. jigsaw.mesh.
        expected = obj.__module__  # type: ignore[attr-defined]
    assert obj.__module__ == expected  # type: ignore[attr-defined]
    assert not any(part.startswith("_") for part in expected.split("."))

    target: object = importlib.import_module(expected)
    for part in obj.__qualname__.split("."):  # type: ignore[attr-defined]
        target = getattr(target, part)
    assert target is obj


@pytest.mark.parametrize("sub", SUBPACKAGES)
def test_subpackage_exports_are_lifted(sub: str) -> None:
    """Every module a subpackage exports is in ``jigsaw.modules``, every function in ``functional``."""
    module = importlib.import_module(f"jigsaw.modules.{sub}")
    flat = set(jigsaw.modules.__all__)
    functional = set(jigsaw.modules.functional.__all__)
    for name in module.__all__:
        obj = getattr(module, name)
        if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
            assert name in flat, f"{sub}.{name} is not lifted to jigsaw.modules"
        elif inspect.isfunction(obj):
            assert name in functional, (
                f"{sub}.{name} is not in jigsaw.modules.functional"
            )


def test_internal_modules_are_private() -> None:
    """Below ``jigsaw.modules.<sub>``, every module path has a ``_`` component.

    Only ``jigsaw.modules``, ``jigsaw.modules.functional`` and the subpackages
    themselves are public; anything else must be marked internal so it can move.
    """
    allowed = {"jigsaw.modules.functional"} | {
        f"jigsaw.modules.{sub}" for sub in SUBPACKAGES
    }
    public_looking = [
        info.name
        for info in pkgutil.walk_packages(
            jigsaw.modules.__path__, prefix="jigsaw.modules."
        )
        if info.name not in allowed
        and not any(part.startswith("_") for part in info.name.split(".")[2:])
    ]
    assert not public_looking
