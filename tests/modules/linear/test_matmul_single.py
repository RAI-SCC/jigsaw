"""Single-process (parallelism 1) checks of the matmul modules and autograd functions.

At parallelism 1 the kernels are plain ``torch`` matmuls, so the distributed matmul
modules can be compared bit-for-bit against ``torch`` autograd without any
communication. That exercises the ``XWT`` / ``XW`` / ``XTW`` forward, backward and
``vmap`` rules and the parameter-free ``MatMul*`` modules on every rank.
"""

import pytest
import torch

from jigsaw.modules.linear._linear import (
    DistributedMatMulXTW,
    DistributedMatMulXW,
    DistributedMatMulXWT,
    MatMulXTW,
    MatMulXW,
    MatMulXWT,
)

CPU = torch.device("cpu")

#: (distributed module, plain module, x shape, w shape, reference op)
OPS = {
    "xwt": (
        DistributedMatMulXWT,
        MatMulXWT,
        (2, 4, 5),
        (2, 3, 5),
        lambda x, w: x @ w.mT,
    ),
    "xw": (DistributedMatMulXW, MatMulXW, (2, 4, 5), (2, 5, 3), lambda x, w: x @ w),
    "xtw": (
        DistributedMatMulXTW,
        MatMulXTW,
        (2, 5, 4),
        (2, 5, 3),
        lambda x, w: x.mT @ w,
    ),
}


def _distributed(cls):
    return cls(device=CPU, model_group=None)


def _operands(x_shape, w_shape):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(*x_shape, dtype=torch.double, generator=g, requires_grad=True)
    w = torch.randn(*w_shape, dtype=torch.double, generator=g, requires_grad=True)
    return x, w


@pytest.mark.parametrize("op", OPS)
class TestSingleProcessMatMul:
    def test_forward_and_backward_match_torch(self, op):
        dist_cls, _, xs, ws, ref = OPS[op]
        x, w = _operands(xs, ws)
        x_ref, w_ref = (
            x.detach().clone().requires_grad_(),
            w.detach().clone().requires_grad_(),
        )

        out = _distributed(dist_cls)(x, w)
        expected = ref(x_ref, w_ref)
        assert torch.allclose(out, expected)

        grad = torch.randn_like(expected)
        out.backward(grad)
        expected.backward(grad)
        assert torch.allclose(x.grad, x_ref.grad)
        assert torch.allclose(w.grad, w_ref.grad)

    def test_vmap_over_leading_dimension(self, op):
        dist_cls, _, xs, ws, ref = OPS[op]
        module = _distributed(dist_cls)
        g = torch.Generator().manual_seed(1)
        x = torch.randn(3, *xs, dtype=torch.double, generator=g)
        w = torch.randn(3, *ws, dtype=torch.double, generator=g)
        got = torch.vmap(module)(x, w)
        want = torch.stack([ref(x[i], w[i]) for i in range(3)])
        assert torch.allclose(got, want)

    def test_parameter_count_is_zero_and_plain_module_matches(self, op):
        dist_cls, plain_cls, xs, ws, ref = OPS[op]
        assert _distributed(dist_cls).get_parameter_count() == 0
        plain = plain_cls()
        assert plain.get_parameter_count() == 0
        x, w = _operands(xs, ws)
        assert torch.equal(plain(x, w), ref(x, w))

    def test_convert_to_jigsaw_reads_the_model_group(self, op):
        dist_cls, plain_cls, *_ = OPS[op]

        class Group:
            def rank(self):
                return 0

            def size(self):
                return 1

        group = Group()
        converted = dist_cls.convert_to_jigsaw(plain_cls(), {"model": group}, CPU)
        assert isinstance(converted, dist_cls)
        assert converted.model_group is group

    def test_local_rank_is_ignored_with_a_warning(self, op):
        dist_cls, _, xs, ws, ref = OPS[op]
        with pytest.warns(DeprecationWarning, match="local_rank"):
            module = dist_cls(device=CPU, model_group=None, local_rank=3)
        x, w = _operands(xs, ws)
        assert torch.allclose(module(x, w), ref(x, w))
