"""``DistributedLinear`` under ``torch.vmap``, compared to a per-member loop.

Layers are built without any flag; the batching rule of ``_XWT`` detects
vmapped weights (stacked ensemble members) at call time.
"""

import pytest
import torch
import torch.distributed as dist
from torch.func import functional_call, stack_module_state

import jigsaw.modules.linear as linear
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]

E = 3
IN, OUT = 4, 8


def _layer(mesh: dict, device: torch.device) -> linear.DistributedLinear:
    return linear.DistributedLinear(
        IN,
        OUT,
        local_rank=mesh["model"].rank(),
        device=device,
        model_group=mesh["model"],
        row_group=mesh["row"],
        col_group=mesh["col"],
        bias=True,
    )


def _check(name: str, got: torch.Tensor, ref: torch.Tensor, device, mesh) -> None:
    assert_close(name, got, ref, 1e-5, 1e-5, device, mesh["model"])


class TestLinearVmap:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "x_shape, x_batched",
        [((2, 5), True), ((2, 5), False), ((5,), True), ((2, 3, 5), True)],
        ids=["bsd", "bsd-shared-x", "sd", "extra-batch-dims"],
    )
    def test_member_weights(
        self, device, process_mesh_dict_2x2, x_shape, x_batched
    ) -> None:
        """Stacked per-member weights: forward and backward match a loop."""
        mesh = process_mesh_dict_2x2
        torch.manual_seed(mesh["model"].rank())
        members = [_layer(mesh, device) for _ in range(E)]
        base = _layer(mesh, device)
        params, _ = stack_module_state(members)

        in_local = IN // mesh["col"].size()
        x = torch.randn(
            E if x_batched else 1, *x_shape, in_local, device=device
        ).requires_grad_()
        x_m = [x[e if x_batched else 0] for e in range(E)]

        def f(p, xi):
            return functional_call(base, p, (xi,))

        got = torch.vmap(f, in_dims=(0, 0 if x_batched else None))(
            params, x if x_batched else x[0]
        )
        ref = torch.stack([m(xi) for m, xi in zip(members, x_m, strict=True)])
        _check("output", got, ref, device, mesh)

        g = torch.randn_like(ref)
        (got * g).sum().backward()
        grad_x_got = x.grad.clone()
        x.grad = None
        (ref * g).sum().backward()
        _check("grad_x", grad_x_got, x.grad, device, mesh)

        _check(
            "grad_weight",
            params["weight"].grad,
            torch.stack([m.weight.grad for m in members]),
            device,
            mesh,
        )
        # functional_call bypasses the bias replica grad-sync; replay it.
        grad_b = params["bias"].grad.clone()
        dist.all_reduce(grad_b, group=mesh["row"])
        _check(
            "grad_bias",
            grad_b,
            torch.stack([m.bias.grad for m in members]),
            device,
            mesh,
        )

    @pytest.mark.dist(4)
    def test_shared_weights(self, device, process_mesh_dict_2x2) -> None:
        """Only the input is vmapped: the axis folds into the batch."""
        mesh = process_mesh_dict_2x2
        torch.manual_seed(mesh["model"].rank())
        layer = _layer(mesh, device)
        x = torch.randn(E, 2, 5, IN // mesh["col"].size(), device=device)

        got = torch.vmap(layer)(x)
        ref = torch.stack([layer(xi) for xi in x])
        _check("output", got, ref, device, mesh)


class TestLinearNestedVmap:
    B = 2  # size of the inner vmap

    def _setup(self, mesh: dict, device: torch.device) -> tuple:
        torch.manual_seed(mesh["model"].rank())
        members = [_layer(mesh, device) for _ in range(E)]
        base = _layer(mesh, device)
        params, _ = stack_module_state(members)
        x = torch.randn(E, self.B, 2, 5, IN // mesh["col"].size(), device=device)
        return members, base, params, x

    @pytest.mark.dist(4)
    def test_outer_member_weights_inner_input(
        self, device, process_mesh_dict_2x2
    ) -> None:
        """Outer vmap over member weights, inner over the input: matches a loop.

        The inner level folds its axis into the batch, so only the outer level
        uses the members config; the bias gradient keeps one members axis.
        """
        mesh = process_mesh_dict_2x2
        members, base, params, x = self._setup(mesh, device)

        def f(p, xe):
            return torch.vmap(lambda xi: functional_call(base, p, (xi,)))(xe)

        got = torch.vmap(f)(params, x)
        ref = torch.stack(
            [torch.stack([m(xi) for xi in x[e]]) for e, m in enumerate(members)]
        )
        _check("output", got, ref, device, mesh)

        g = torch.randn_like(ref)
        (got * g).sum().backward()
        (ref * g).sum().backward()
        _check(
            "grad_weight",
            params["weight"].grad,
            torch.stack([m.weight.grad for m in members]),
            device,
            mesh,
        )
        # functional_call bypasses the bias replica grad-sync; replay it.
        grad_b = params["bias"].grad.clone()
        dist.all_reduce(grad_b, group=mesh["row"])
        _check(
            "grad_bias",
            grad_b,
            torch.stack([m.bias.grad for m in members]),
            device,
            mesh,
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("outer", ["input", "member_weights"])
    def test_inner_member_weights_raises(
        self, device, process_mesh_dict_2x2, outer: str
    ) -> None:
        """A vmap around one over member weights reaches the rule twice: raises."""
        mesh = process_mesh_dict_2x2
        _, base, params, x = self._setup(mesh, device)

        def g(p, xi):
            return functional_call(base, p, (xi,))

        with pytest.raises(NotImplementedError, match="nested"):
            if outer == "input":
                torch.vmap(lambda xb: torch.vmap(g)(params, xb))(x.transpose(0, 1))
            else:
                p2 = {
                    k: v.unsqueeze(1).expand(E, self.B, *v.shape[1:]).contiguous()
                    for k, v in params.items()
                }
                torch.vmap(torch.vmap(g))(p2, x)
