"""jigsaw modules wrapped in ``DistributedDataParallel`` over the ``dp`` axis.

Every replicated parameter must end a backward pass with the gradient
``AVG_dp(SUM_replica(local))`` and be bitwise identical across its replica
groups; every sharded parameter with ``AVG_dp(local)``. ``local`` is the
gradient the same rank gets with the in-graph sync switched off, so the
expected value is built from per-rank quantities by explicit collectives,
independently of the code under test.

The linear and conv cells use small integer data, so every sum is exact and the
comparison is ``torch.equal``. Attention goes through a softmax and layer norm
through a square root; both are compared to a tight tolerance, replica identity
is still bitwise.
"""

import contextlib

import pytest
import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel

from jigsaw import parallel
from jigsaw.mesh import make_process_mesh
from jigsaw.modules import convolution, linear, normalization
from jigsaw.modules.attention import attention
from jigsaw.parallel import _sync

# (dp, row, col) -> world size
MESHES = [(2, 2, 2), (2, 2, 1), (4, 1, 1), (1, 2, 2)]
KINDS = ["xwt", "conv_halo", "conv_patchify", "attention", "layer_norm"]
VARIANTS = [
    "no_ddp",
    "ddp",
    "bucket_view",
    "find_unused",
    "static_graph",
    "fp16_hook",
    "no_sync",
    "device_mesh",
    "joint",
    "joint_bucket_view",
    "joint_static_graph",
    "joint_fp16",
    "joint_no_sync",
    "joint_accum",
]
EXACT_KINDS = {"xwt", "conv_halo", "conv_patchify"}

_MESH_CACHE: dict = {}


def _mesh(shape: tuple[int, int, int], device: torch.device) -> dict:
    if shape not in _MESH_CACHE:
        dp, row, col = shape
        _MESH_CACHE[shape] = make_process_mesh(
            {"dp": dp, "row": row, "col": col},
            device_type=device.type,
            model_parallel_axes=("row", "col"),
        )
    return _MESH_CACHE[shape]


def _int_tensor(shape, seed: int, device, dtype=torch.float32) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(-2, 3, shape, generator=g).to(device=device, dtype=dtype)


def _build(kind: str, mesh: dict, device: torch.device, fill: bool = True):
    """Return ``(model, make_input)`` for one module kind on ``mesh``.

    With ``fill`` every parameter is overwritten with the same small integers on
    every rank; without it the parameters keep their initialization.
    """
    row, col = mesh["row"], mesh["col"]
    if kind == "xwt":
        summa = (row.size(), col.size()) in {(1, 1), (1, 2), (2, 2)}
        model = linear.DistributedLinear(
            8,
            8,
            local_rank=(mesh["model"] if summa else col).rank(),
            device=device,
            model_group=mesh["model"] if summa else col,
            col_group=col,
            row_group=row,
            bias=True,
        )
        if not summa:
            # ``row`` is a pure replication axis here, as in attention.
            parallel.mark_replicated(model.weight, row)
            dist.broadcast(model.weight.data, group_src=0, group=row)
        shape = (2, 4 // row.size() if summa else 4, 8 // col.size())
    elif kind == "conv_halo":
        model = convolution.DistributedConv2d(
            4, 4, 3, device=device, spatial_group=row, channel_group=col, padding=(1, 1)
        )
        shape = (2, 4 // col.size(), 4, 8 // row.size())
    elif kind == "conv_patchify":
        model = convolution.DistributedConv2d(
            4, 4, 2, device=device, spatial_group=row, channel_group=col, stride=2
        )
        shape = (2, 4 // col.size(), 4, 8 // row.size())
    elif kind == "attention":
        model = attention.DistributedAttention(
            8,
            8,
            8,
            heads=2,
            dropout_p=0.0,
            local_rank=col.rank(),
            device=device,
            spatial_group=row,
            channel_group=col,
            qk_norm=True,
        )
        shape = (2, 8 // row.size(), 8 // col.size())
    elif kind == "layer_norm":
        model = normalization.DistributedLayerNorm(
            8,
            local_rank=mesh["model"].rank(),
            device=device,
            row_group=row,
            col_group=col,
        )
        shape = (2, 8 // row.size(), 8 // col.size())
    else:
        raise ValueError(kind)

    for i, p in enumerate(model.parameters() if fill else ()):
        p.data = _int_tensor(p.shape, 1000 + i, device, p.dtype)

    def make_input(step: int) -> torch.Tensor:
        seed = 10 * dist.get_rank() + step
        if kind in EXACT_KINDS:
            return _int_tensor(shape, seed, device)
        g = torch.Generator().manual_seed(seed)
        return torch.randn(shape, generator=g).to(device)

    return model, make_input


def _wrap(model: torch.nn.Module, variant: str, dp: dist.ProcessGroup, mesh=None):
    if variant == "no_ddp":
        return model
    if variant == "device_mesh":
        # A 1-D slice of the n-d DeviceMesh instead of ``process_group=``.
        return DistributedDataParallel(model, device_mesh=mesh.device_mesh["dp"])
    joint = variant.startswith("joint")
    base = variant.removeprefix("joint_") if joint else variant
    kwargs = {
        "bucket_view": {"gradient_as_bucket_view": True},
        "find_unused": {"find_unused_parameters": True},
        "static_graph": {"static_graph": True},
    }.get(base, {})
    ddp = DistributedDataParallel(model, process_group=dp, **kwargs)
    if variant == "fp16_hook":
        ddp.register_comm_hook(dp, default_hooks.fp16_compress_hook)
    if joint:
        wrapper = default_hooks.fp16_compress_wrapper if base == "fp16" else None
        parallel.register_joint_comm_hook(ddp, wrapper)
    return ddp


def _n_micro(variant: str) -> int:
    """Microbatches per step: 2 for the accumulation variants."""
    return 2 if variant in ("no_sync", "joint_no_sync", "joint_accum") else 1


def _uses_no_sync(variant: str) -> bool:
    return variant in ("no_sync", "joint_no_sync")


def _step(model, make_input, n_micro: int, no_sync: bool = False) -> None:
    for step in range(n_micro):
        ctx = (
            model.no_sync()
            if no_sync and step < n_micro - 1
            else contextlib.nullcontext()
        )
        with ctx:
            y = model(make_input(step))
            gout = _int_tensor(y.shape, 500 + 10 * dist.get_rank() + step, y.device)
            y.backward(gout)


def _disable_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``sync_grad`` a pass-through, as ``JIGSAW_DISABLE_GRAD_SYNC=1`` does."""
    # Patch the module that reads the switch; jigsaw.parallel only re-exports.
    monkeypatch.setattr(_sync, "_SYNC_DISABLED", True)


def _local_grads(model, make_input, n_micro: int, monkeypatch) -> list:
    with monkeypatch.context() as m:
        _disable_sync(m)
        _step(model, make_input, n_micro)
    grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    return grads


def _expected(model, local: list, dp: dist.ProcessGroup, ddp: bool) -> list:
    out = []
    for p, g in zip(model.parameters(), local):
        e = g.clone()
        for grp in parallel.replica_groups(p):
            dist.all_reduce(e, group=grp)
        if ddp and dp.size() > 1:
            e /= dp.size()
            dist.all_reduce(e, group=dp)
        out.append(e)
    return out


def _assert_identical_over(t: torch.Tensor, group: dist.ProcessGroup, what: str):
    gathered = [torch.empty_like(t) for _ in range(group.size())]
    dist.all_gather(gathered, t.contiguous(), group=group)
    ok = all(torch.equal(g, gathered[0]) for g in gathered)
    assert ok, f"{what} differs across {dist.get_process_group_ranks(group)}"


def _cases():
    for shape in MESHES:
        world = shape[0] * shape[1] * shape[2]
        for kind in KINDS:
            for variant in VARIANTS:
                yield pytest.param(
                    shape,
                    kind,
                    variant,
                    id=f"{'x'.join(map(str, shape))}-{kind}-{variant}",
                    marks=pytest.mark.dist(world),
                )


@pytest.mark.parametrize("shape, kind, variant", list(_cases()))
def test_ddp_composition(shape, kind, variant, device, monkeypatch) -> None:
    mesh = _mesh(shape, device)
    dp = mesh["dp"]
    model, make_input = _build(kind, mesh, device)
    n_micro = _n_micro(variant)

    local = _local_grads(model, make_input, n_micro, monkeypatch)
    expected = _expected(model, local, dp, ddp=variant != "no_ddp")

    wrapped = _wrap(model, variant, dp, mesh)
    _step(wrapped, make_input, n_micro, no_sync=_uses_no_sync(variant))

    for (name, p), e in zip(model.named_parameters(), expected):
        assert p.grad is not None, name
        if kind in EXACT_KINDS:
            assert torch.equal(p.grad, e), f"{name}: {p.grad} != {e}"
        else:
            # The fp16 comm hook rounds non-integer gradients to fp16. In fp32,
            # summing microbatches before or after the replica all-reduce
            # differs by ~1e-5 of the largest entry (cancellation in the
            # softmax backward); a wrong reduction is off by a factor.
            rtol, atol = (
                (1e-2, max(2e-3 * e.abs().max().item(), 1e-6))
                if variant in ("fp16_hook", "joint_fp16")
                else (1e-5, max(1e-4 * e.abs().max().item(), 1e-6))
            )
            torch.testing.assert_close(
                p.grad, e, rtol=rtol, atol=atol, msg=lambda m: f"{name}: {m}"
            )
        for grp in parallel.replica_groups(p):
            _assert_identical_over(p.grad, grp, f"{name}.grad")
        if variant != "no_ddp":
            _assert_identical_over(p.grad, dp, f"{name}.grad")


@pytest.mark.dist(4)
def test_sync_switched_off_is_detected(device, monkeypatch) -> None:
    """The matrix above fails if the in-graph reduction does not happen."""
    mesh = _mesh((2, 2, 1), device)
    model, make_input = _build("conv_halo", mesh, device)
    local = _local_grads(model, make_input, 1, monkeypatch)
    expected = _expected(model, local, mesh["dp"], ddp=True)
    wrapped = _wrap(model, "ddp", mesh["dp"])
    _disable_sync(monkeypatch)
    _step(wrapped, make_input, 1)
    assert not all(torch.equal(p.grad, e) for p, e in zip(model.parameters(), expected))


@pytest.mark.dist(8)
def test_joint_hook_groups(device) -> None:
    """Each parameter is averaged over exactly the ranks that hold a copy of it."""
    mesh = _mesh((2, 2, 2), device)
    model, _ = _build("xwt", mesh, device)  # weight sharded; bias replicated over row
    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    state = parallel.register_joint_comm_hook(ddp)
    dp_ranks = set(dist.get_process_group_ranks(mesh["dp"]))
    assert any(parallel.replica_groups(p) for p in model.parameters())
    for name, p in model.named_parameters():
        got = set(dist.get_process_group_ranks(state.group_of(p)))
        assert dist.get_rank() in got, name
        if parallel.replica_groups(p):
            row = set(dist.get_process_group_ranks(mesh["row"]))
            # dp x row: both axes, and nothing along col
            assert len(got) == mesh["dp"].size() * mesh["row"].size(), (name, got)
            assert dp_ranks <= got and row <= got, (name, got)
        else:
            assert got == dp_ranks, (name, got)


@pytest.mark.dist(4)
@pytest.mark.parametrize("reducer", ["cpp", "python"])
def test_joint_state_without_hook_raises(device, reducer, request) -> None:
    """Scaling without the hook would leave the gradients unsummed: refuse."""
    if reducer == "python":
        request.getfixturevalue("python_reducer")
    mesh = _mesh((2, 2, 1), device)
    model, make_input = _build("conv_halo", mesh, device)
    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    parallel.JointCommHookState(ddp)
    with pytest.raises(RuntimeError, match="not registered"):
        ddp(make_input(0))


@pytest.fixture
def python_reducer(monkeypatch):
    """Build DDPs with the Python reducer, as for compiled autograd.

    Such a DDP also turns on inductor's DDP fusion and inlines DDP into dynamo,
    process-wide; both are undone afterwards.
    """
    import torch._dynamo
    import torch._inductor.config as inductor_config
    from torch._dynamo import trace_rules

    monkeypatch.setattr(torch._dynamo.config, "optimize_ddp", "python_reducer")
    for name in ("_fuse_ddp_communication", "_fuse_ddp_bucket_size"):
        monkeypatch.setattr(inductor_config, name, getattr(inductor_config, name))
    inline = set(trace_rules.LEGACY_MOD_INLINELIST)
    torch._dynamo.reset()
    yield
    trace_rules.LEGACY_MOD_INLINELIST.clear()
    trace_rules.LEGACY_MOD_INLINELIST.update(inline)
    trace_rules.get_legacy_mod_inlinelist.cache_clear()
    torch._dynamo.reset()


def _compiled_step(model, make_input, n_micro, no_sync, backend: str) -> None:
    """``_step`` with the backward under compiled autograd."""
    from torch._dynamo import compiled_autograd

    with compiled_autograd._enable(torch.compile(backend=backend)):
        _step(model, make_input, n_micro, no_sync)


def _compiled_cases():
    for shape in [(2, 2, 1), (2, 2, 2)]:
        world = shape[0] * shape[1] * shape[2]
        for kind in KINDS:
            for variant in ["compiled", "joint_compiled", "joint_compiled_no_sync"]:
                yield pytest.param(
                    shape,
                    kind,
                    variant,
                    id=f"{'x'.join(map(str, shape))}-{kind}-{variant}",
                    marks=pytest.mark.dist(world),
                )


@pytest.mark.parametrize("shape, kind, variant", list(_compiled_cases()))
def test_ddp_compiled_autograd(
    shape, kind, variant, device, monkeypatch, python_reducer
) -> None:
    """DDP's Python reducer, with and without the joint hook, as in the matrix."""
    mesh = _mesh(shape, device)
    dp = mesh["dp"]
    model, make_input = _build(kind, mesh, device)
    no_sync = variant.endswith("no_sync")
    n_micro = 2 if no_sync else 1

    local = _local_grads(model, make_input, n_micro, monkeypatch)
    expected = _expected(model, local, dp, ddp=True)

    ddp = DistributedDataParallel(model, process_group=dp)
    assert ddp._use_python_reducer
    if variant.startswith("joint"):
        parallel.register_joint_comm_hook(ddp)
    _compiled_step(ddp, make_input, n_micro, no_sync, "aot_eager")

    for (name, p), e in zip(model.named_parameters(), expected):
        assert p.grad is not None, name
        if kind in EXACT_KINDS:
            assert torch.equal(p.grad, e), f"{name}: {p.grad} != {e}"
        else:
            torch.testing.assert_close(
                p.grad,
                e,
                rtol=1e-5,
                atol=max(1e-4 * e.abs().max().item(), 1e-6),
                msg=lambda m: f"{name}: {m}",
            )
        for grp in [*parallel.replica_groups(p), dp]:
            _assert_identical_over(p.grad, grp, f"{name}.grad")


@pytest.mark.dist(4)
def test_joint_hook_inductor_fusion(device, monkeypatch, python_reducer) -> None:
    """Inductor's DDP fusion must not merge all-reduces over different groups.

    It fuses every ``div -> all_reduce -> copy_`` in the backward into one
    all-reduce over the last one's group; a replicated conv (joint group dp x
    row) next to an unmarked layer (dp) would be reduced over one group.
    """
    mesh = _mesh((2, 2, 1), device)
    conv, make_input = _build("conv_halo", mesh, device)
    plain = torch.nn.Conv2d(4, 4, 1, device=device)
    for i, p in enumerate(plain.parameters()):
        p.data = _int_tensor(p.shape, 2000 + i, device, p.dtype)
    model = torch.nn.Sequential(conv, plain)

    local = _local_grads(model, make_input, 1, monkeypatch)
    expected = _expected(model, local, mesh["dp"], ddp=True)

    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    parallel.register_joint_comm_hook(ddp)
    _compiled_step(ddp, make_input, 1, False, "inductor")

    for (name, p), e in zip(model.named_parameters(), expected):
        assert torch.equal(p.grad, e), f"{name}: {p.grad} != {e}"


@pytest.mark.dist(4)
def test_joint_hook_python_reducer_rejects_wrapper(device, python_reducer) -> None:
    """Compression wrappers need DDP's buckets, which the Python reducer lacks."""
    mesh = _mesh((2, 2, 1), device)
    model, _ = _build("conv_halo", mesh, device)
    ddp = DistributedDataParallel(model, process_group=mesh["dp"])
    with pytest.raises(ValueError, match="Python reducer"):
        parallel.register_joint_comm_hook(ddp, default_hooks.fp16_compress_wrapper)
