"""Kernel selection in ``LinearConfig`` (pure dispatch logic, no communication)."""

import pytest
import torch

import jigsaw.modules.linear._distributed_matmul as dist_mm
import jigsaw.modules.linear._distributed_matmul_p2p as dist_mm_p2p
from jigsaw.modules.linear._config import LinearConfig

CPU, CUDA = torch.device("cpu"), torch.device("cuda")
WAYS = {2: "twoway", 4: "fourway", 8: "eightway"}
BIAS = torch.zeros(3)


def _config(parallelism=1, bias=BIAS, linear_layer=True, device=CPU):
    return LinearConfig(parallelism, bias, linear_layer=linear_layer, device=device)


class TestMatmulKernelSelection:
    @pytest.mark.parametrize(
        "dev, module", [(CPU, dist_mm_p2p), (CUDA, dist_mm)], ids=["cpu", "cuda"]
    )
    @pytest.mark.parametrize("parallelism, way", WAYS.items())
    def test_distributed_kernels_come_from_the_device_backend(
        self, dev, module, parallelism, way
    ):
        xwt, xw, xtw, fused = _config(parallelism, device=dev).mm_kernels
        assert xwt is getattr(module, f"xw_transpose_{way}")
        assert xw is getattr(module, f"xw_{way}")
        assert xtw is getattr(module, f"x_transpose_w_{way}")
        assert fused is getattr(module, f"fused_xw_xtw_{way}")

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
    @pytest.mark.parametrize("parallelism", [0, 3, 16])
    def test_unsupported_parallelism_is_rejected(self, dev, parallelism):
        with pytest.raises(ValueError, match="Unsupported parallelism"):
            _config(parallelism, device=dev)

    @pytest.mark.parametrize("parallelism", [1, 2])
    def test_unsupported_device_is_rejected(self, parallelism):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            _config(parallelism, device=torch.device("meta"))


SELECTORS = [
    "_select_xw_transpose_kernel",
    "_select_xw_kernel",
    "_select_x_transpose_w_kernel",
    "_select_xw_xtw_fused_kernel",
]


class TestForVmap:
    def test_is_cached_and_idempotent(self):
        cfg = _config()
        vc = cfg.for_vmap()
        assert vc is cfg.for_vmap() and vc.for_vmap() is vc
        assert vc.vmapped and not cfg.vmapped


class TestEachSelectorValidates:
    """``_mm_kernel_selection`` stops at the first bad selector, so hit each directly."""

    @pytest.mark.parametrize("selector", SELECTORS)
    @pytest.mark.parametrize("dev", [CPU, CUDA], ids=["cpu", "cuda"])
    def test_unsupported_parallelism(self, selector, dev):
        with pytest.raises(ValueError, match="Unsupported parallelism: 3"):
            getattr(_config(), selector)(3, dev)

    @pytest.mark.parametrize("selector", SELECTORS)
    def test_unsupported_device(self, selector):
        with pytest.raises(ValueError, match="Unsupported device meta"):
            getattr(_config(), selector)(2, torch.device("meta"))
