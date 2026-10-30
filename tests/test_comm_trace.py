"""Tests for the communication tracer in jigsaw._core.comm_trace.

Only the two-way kernels label their sends. The batched point-to-point kernels run
for real on the session device, on two-way model groups with 1, 2 or 4
data-parallel replicas. Gloo has no ``all_to_all``, so the all-to-all kernels run
on one process against a recording fake of ``dist.all_to_all`` and a fake group;
their numerical output is meaningless there, only the recorded labels are checked.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.linear as linear
from jigsaw._core import comm_trace
from jigsaw.modules.linear import _distributed_matmul as dist_mm
from tests.parity import all_ranks_ok, assert_close
from tests.process_mesh import make_process_mesh

#: Payload kind each two-way kernel sends.
_KINDS = {
    "xw_transpose_twoway": "partial",
    "xw_twoway": "x",
    "x_transpose_w_twoway": "x",
    "fused_xw_xtw_twoway": "grad_output",
}
#: Backward kernel of each layer of :func:`_model`: layer 0 has no input
#: gradient, layer 1 a frozen weight, layer 2 both gradients.
_BACKWARD_KERNELS = {
    "0": "x_transpose_w_twoway",
    "1": "xw_twoway",
    "2": "fused_xw_xtw_twoway",
}
_MESHES = [
    pytest.param("process_mesh_dict_1x2", marks=pytest.mark.dist(2), id="1x2"),
    pytest.param("process_mesh_dict_2x1x2", marks=pytest.mark.dist(4), id="2x1x2"),
    pytest.param("mesh_4x1x2", marks=pytest.mark.dist(8), id="4x1x2"),
]


@pytest.fixture(scope="session")
def mesh_4x1x2(device: torch.device) -> dict:
    """Two-way model groups on four data-parallel replicas (world 8)."""
    return make_process_mesh(
        device_type=device.type,
        mesh_shape=[4, 1, 2],
        mesh_dim_names=["dp", "row", "col"],
    )


def _model(mesh: dict, device: torch.device) -> torch.nn.Sequential:
    """Build three distributed linear layers whose backward uses three kernels."""
    torch.manual_seed(0)
    layers = [
        linear.DistributedLinear(
            8,
            8,
            local_rank=mesh["model"].rank(),
            device=device,
            model_group=mesh["model"],
            row_group=mesh["row"],
            col_group=mesh["col"],
            bias=False,
        )
        for _ in range(3)
    ]
    layers[1].weight.requires_grad_(False)
    return torch.nn.Sequential(*layers)


def _input(device: torch.device) -> torch.Tensor:
    """Return this rank's tile of a ``(2, 8, 8)`` input split over two features."""
    torch.manual_seed(1)
    return torch.randn(2, 8, 4, device=device)


class TestDistributed:
    @pytest.mark.parametrize("mesh_name", _MESHES)
    def test_labels(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Every send carries its layer, direction, kernel, kind and size, every step."""
        mesh = request.getfixturevalue(mesh_name)
        group = mesh["model"]
        model, x = _model(mesh, device), _input(device)
        with comm_trace.trace(model) as tracer:
            for _ in range(2):
                model(x).sum().backward()
                tracer.next_step()

        # Every two-way payload is an output-sized tile sent to the other rank.
        errors = []
        for r in tracer.records:
            if r.kernel not in _KINDS or r.kind != _KINDS[r.kernel]:
                errors.append(f"unexpected kernel or kind: {r}")
            if r.peer != 1 - group.rank():
                errors.append(f"bad peer: {r}")
            if r.shape != (2, 8, 4) or r.nbytes != 2 * 8 * 4 * r.dtype.itemsize:
                errors.append(f"bad size: {r}")
        expected = [(layer, "forward", "xw_transpose_twoway") for layer in "012"]
        expected += [(layer, "backward", k) for layer, k in _BACKWARD_KERNELS.items()]
        for step in range(2):
            got = [
                (r.layer, r.direction, r.kernel)
                for r in tracer.records
                if r.step == step
            ]
            if sorted(got) != sorted(expected):
                errors.append(f"step {step}: {got}")
        step_0 = [r for r in tracer.records if r.step == 0]
        step_1 = [dataclasses.replace(r, step=0) for r in tracer.records if r.step == 1]
        if step_0 != step_1:
            errors.append("steps 0 and 1 sent different payloads")
        if tracer.total_bytes() != sum(r.nbytes for r in tracer.records):
            errors.append("total_bytes does not sum the records")
        assert all_ranks_ok(not errors, device, group), errors

    @pytest.mark.parametrize("mesh_name", _MESHES)
    def test_hook(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """The hook sees every record together with its payload."""
        mesh = request.getfixturevalue(mesh_name)
        group = mesh["model"]
        model, x = _model(mesh, device), _input(device)
        hooked = []
        with comm_trace.trace(
            model, hook=lambda r, t: hooked.append((r, t.clone()))
        ) as tracer:
            model(x).sum().backward()

        errors = []
        if [r for r, _ in hooked] != tracer.records:
            errors.append("hook records differ from tracer.records")
        if any(tuple(t.shape) != r.shape or t.dtype != r.dtype for r, t in hooked):
            errors.append("payload shape or dtype differs from its record")
        # ``sum()`` makes the last layer's ``grad_output`` all ones.
        last = [t for r, t in hooked if r.layer == "2" and r.kind == "grad_output"]
        if len(last) != 1 or not torch.equal(last[0], torch.ones_like(last[0])):
            errors.append("last layer's grad_output payload is not all ones")
        assert all_ranks_ok(not errors, device, group), errors

    @pytest.mark.parametrize("mesh_name", _MESHES)
    def test_tracing_does_not_change_results(
        self, request: pytest.FixtureRequest, device: torch.device, mesh_name: str
    ) -> None:
        """Outputs and gradients are bitwise identical with tracing and a hook on."""
        mesh = request.getfixturevalue(mesh_name)
        group = mesh["model"]
        model, x = _model(mesh, device), _input(device)

        def run() -> tuple[torch.Tensor, list[torch.Tensor]]:
            model.zero_grad()
            out = model(x)
            out.sum().backward()
            grads = [p.grad.clone() for p in model.parameters() if p.grad is not None]
            return out.detach(), grads

        out_ref, grads_ref = run()
        with comm_trace.trace(model, hook=lambda r, t: t.clone()):
            out, grads = run()
        assert_close("output", out, out_ref, 0.0, 0.0, device, group)
        for i, (g, g_ref) in enumerate(zip(grads, grads_ref, strict=True)):
            assert_close(f"grad {i}", g, g_ref, 0.0, 0.0, device, group)


class _FakeGroup:
    def __init__(self, rank: int) -> None:
        self._rank = rank

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return 2


@pytest.mark.parametrize("kernel_name", list(_KINDS))
def test_all_to_all_kernel_labels(
    monkeypatch: pytest.MonkeyPatch, kernel_name: str
) -> None:
    """Each two-way all-to-all kernel labels its send with its name and kind."""
    handle = SimpleNamespace(wait=lambda: None)
    monkeypatch.setattr(dist, "all_to_all", lambda *args, **kwargs: handle)
    kernel = getattr(dist_mm, kernel_name)
    # A 1x2 tile layout with 16 features in and out: every operand is 16 x 8.
    n_args = 3 if kernel_name.startswith("fused") else 2
    args = [torch.randn(16, 8) for _ in range(n_args)]
    for rank in range(2):
        with comm_trace.trace() as tracer:
            kernel(*args, rank, torch.device("cpu"), _FakeGroup(rank))
        (record,) = tracer.records
        assert (record.kernel, record.kind) == (kernel_name, _KINDS[kernel_name])
        assert (record.peer, record.shape) == (1 - rank, (16, 8))


def test_labels_inside_scope_and_kernel() -> None:
    """``scope`` and ``kernel`` label the sends issued inside them."""

    @comm_trace.kernel
    def send() -> None:
        comm_trace.record_send(torch.zeros(3, 2), "w", None, group_dst=1)

    with comm_trace.trace() as tracer:
        send()
        with comm_trace.scope("layer", "backward"):
            send()
    first, second = tracer.records
    assert (first.layer, first.direction, first.kernel) == (None, None, "send")
    assert (second.layer, second.direction, second.kind) == ("layer", "backward", "w")
    assert (second.peer, second.shape, second.nbytes) == (1, (3, 2), 24)
    assert tracer.total_bytes() == 48


def test_off_records_nothing() -> None:
    """Without an active trace the hooks are no-ops."""
    assert comm_trace.layer_name(torch.nn.Linear(1, 1)) is None
    with comm_trace.scope("layer", "forward"):
        assert comm_trace.current_layer() is None
    comm_trace.record_send(torch.zeros(1), "x", None, group_dst=0)


def test_nested_trace_raises_and_error_resets() -> None:
    """A second trace raises, and an error inside a trace ends it."""
    with comm_trace.trace(), pytest.raises(RuntimeError, match="already active"):
        with comm_trace.trace():
            pass
    with pytest.raises(ValueError), comm_trace.trace():
        raise ValueError
    assert comm_trace._tracer is None
