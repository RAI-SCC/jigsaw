"""Kernel selection in ``LinearConfig`` (pure dispatch logic, no communication)."""

import pytest
import torch

import jigsaw.modules.linear._distributed_matmul as dist_mm
import jigsaw.modules.linear._distributed_matmul_batchedp2p as dist_mm_batchedp2p
import jigsaw.modules.linear._distributed_matmul_p2p as dist_mm_p2p
from jigsaw.modules.linear._config import LinearConfig

CPU, CUDA = torch.device("cpu"), torch.device("cuda")
WAYS = {2: "twoway", 4: "fourway", 8: "eightway"}


class _Group:
    """Stand-in for a process group; kernel selection only reads its size."""

    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


def _config(size=1, linear_layer=True, device=CPU, comm="auto"):
    return LinearConfig(
        _Group(size), linear_layer=linear_layer, device=device, comm=comm
    )


class TestMatmulKernelSelection:
    @pytest.mark.parametrize(
        "dev, module", [(CPU, dist_mm_p2p), (CUDA, dist_mm)], ids=["cpu", "cuda"]
    )
    @pytest.mark.parametrize("size, way", WAYS.items())
    def test_distributed_kernels_come_from_the_device_backend(
        self, dev, module, size, way
    ):
        xwt, xw, xtw, fused = _config(size, device=dev).mm_kernels
        assert xwt is getattr(module, f"xw_transpose_{way}")
        assert xw is getattr(module, f"xw_{way}")
        assert xtw is getattr(module, f"x_transpose_w_{way}")
        assert fused is getattr(module, f"fused_xw_xtw_{way}")

    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    @pytest.mark.parametrize(
        "comm, module",
        [
            ("all_to_all", dist_mm),
            ("p2p", dist_mm_p2p),
            ("batched_p2p", dist_mm_batchedp2p),
        ],
    )
    @pytest.mark.parametrize("size, way", WAYS.items())
    def test_explicit_comm_overrides_the_device_default(
        self, dev, comm, module, size, way
    ):
        cfg = _config(size, device=dev, comm=comm)
        assert cfg.comm == comm
        xwt, xw, xtw, fused = cfg.mm_kernels
        assert xwt is getattr(module, f"xw_transpose_{way}")
        assert xw is getattr(module, f"xw_{way}")
        assert xtw is getattr(module, f"x_transpose_w_{way}")
        assert fused is getattr(module, f"fused_xw_xtw_{way}")

    @pytest.mark.parametrize(
        "dev, comm", [(CPU, "p2p"), (CUDA, "all_to_all")], ids=["cpu", "cuda"]
    )
    def test_auto_comm_resolves_per_device(self, dev, comm):
        assert _config(2, device=dev).comm == comm

    def test_unknown_comm_is_rejected(self):
        with pytest.raises(ValueError, match="comm must be 'auto' or one of"):
            _config(2, comm="nccl")

    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    def test_single_process_kernels_are_plain_matmuls(self, dev):
        xwt, xw, xtw, fused = _config(1, device=dev).mm_kernels
        x, w = torch.randn(2, 4, 5), torch.randn(3, 5)
        assert torch.equal(xwt(x, w, "ignored", key="ignored"), x @ w.mT)
        w = torch.randn(5, 3)
        assert torch.equal(xw(x, w), x @ w)
        x, w = torch.randn(2, 5, 4), torch.randn(2, 5, 3)
        assert torch.equal(xtw(x, w), x.mT @ w)
        assert fused is dist_mm.fused_xw_xtw_oneway

    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    @pytest.mark.parametrize("size", [0, 3, 16])
    def test_unsupported_model_group_size_is_rejected(self, dev, size):
        with pytest.raises(ValueError, match="Unsupported model group size"):
            _config(size, device=dev)

    def test_no_model_group_means_single_process(self):
        cfg = LinearConfig(None, device=CPU)
        assert cfg.mm_kernels[3] is dist_mm.fused_xw_xtw_oneway

    @pytest.mark.parametrize("size", [1, 2])
    def test_unsupported_device_is_rejected(self, size):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            _config(size, device=torch.device("meta"))


class TestForVmap:
    def test_is_cached_and_idempotent(self):
        cfg = _config()
        vc = cfg.for_vmap()
        assert vc is cfg.for_vmap() and vc.for_vmap() is vc
        assert vc.vmapped and not cfg.vmapped
