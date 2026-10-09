"""
Shared harness for comparing a distributed jigsaw layer against a serial reference.

The pattern: build the serial module, run a forward and backward pass on the *global*
tensors, then split the input and the upstream gradient across the process mesh, run the
distributed module, gather its results back and compare -- output, input gradient,
parameter values and parameter gradients in one call.

Two design-shaping points:

* Backward is driven with an explicit upstream gradient rather than a loss. jigsaw's
  weight-gradient all-reduce (``jigsaw.parallel.sync_grad``) is a ``ReduceOp.SUM``
  over the spatial group, so the distributed gradient only equals the serial one when
  nothing averages over a sharded dimension. Splitting a global ``randn`` upstream
  gradient makes that relation exact and removes any need for a scaling factor.

* Parameters are described as either *sharded* over a mesh axis (compare by gathering)
  or *replicated* over it (compare by checking the replicas agree, then taking one). For
  a replicated parameter the agreement check is itself the test that the gradient sync
  ran: without the all-reduce the per-rank gradients hold only local contributions and
  differ.

Relation to the rest of the suite
---------------------------------
The mesh comes from the ordinary ``process_mesh_*`` fixtures in ``tests/process_mesh``
and the rank count from an ``@pytest.mark.dist(N)`` marker, exactly like the hand-rolled
reference checks in ``tests/modules/convolution``. ``ShardSpec`` is keyed by mesh-axis
*name*, so the same harness drives a 2-D ``row`` x ``col`` mesh and an N-way spatial mesh
(``sh``/``sw``/``sd`` + ``col``) without change. ``assert_close`` / ``all_ranks_ok`` give
every rank the same verdict so a mismatch fails the job instead of hanging it; those two
are re-used directly by the convolution reference checks.
"""

import contextlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    """
    Disable TF32 for the duration of a parity comparison, restoring it afterwards.

    ``torch.backends.cudnn.allow_tf32`` defaults to True, which gives fp32 convolutions
    a 10-bit mantissa and roughly 1e-3 relative error. That alone would be tolerable if
    both sides made the same error, but the serial and distributed paths contract the
    channel axis in a different order and pick different cuDNN algorithms, so the two
    errors do not cancel. The drift grows with the reduction length (in_channels *
    kernel * kernel), which makes the tolerance a layer-shape property rather than a
    property of the distribution being tested.

    ``tests/conftest.py`` already turns TF32 off session-wide; this keeps the harness
    correct if it is ever imported somewhere that does not.
    """
    cudnn, matmul = (
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.matmul.allow_tf32,
    )
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.backends.cuda.matmul.allow_tf32 = matmul


class ForcedSdpaCalls:
    """Record of what :func:`forced_sdpa_backend` intercepted while it was active.

    Exists so a test can assert the forcing *actually took effect* instead of
    trusting that it did. ``count`` is the number of ``sdpa_kernel`` contexts
    that were redirected, and ``requested`` holds the priority lists the code
    under test asked for, in order, so a mismatch report can say what it
    wanted.
    """

    def __init__(self) -> None:
        self.count = 0
        self.requested: list[Any] = []

    def record(self, backends: Any) -> None:
        self.count += 1
        self.requested.append(backends)


@contextlib.contextmanager
def forced_sdpa_backend(
    backend: torch.nn.attention.SDPBackend,
) -> Iterator[ForcedSdpaCalls]:
    """Force every ``scaled_dot_product_attention`` call onto exactly one torch
    SDPA kernel for the duration of the block, restoring the real dispatcher
    afterwards.

    ``DistributedAttention``'s ``sdpa``/``math`` cores (``_attn_core``,
    ``_attn_core_math``, ``_dense_bshd_core``) each open their *own*
    ``torch.nn.attention.sdpa_kernel([...])`` context with a hardcoded
    priority list (flash first), so a context a caller enters *around* the
    whole call is overridden the moment that inner one is entered --
    nesting ``sdpa_kernel`` does not restrict anything, the innermost one
    always wins. Monkeypatching the ``sdpa_kernel`` symbol itself is the only
    way in: every call site looks it up fresh on ``torch.nn.attention`` each
    time, so replacing it here makes the next kernel dispatch -- whatever
    list it asked for -- resolve to just ``backend``, which then either runs
    or raises if that shape/dtype/mask combination is not one ``backend`` can
    serve. There is no silent fallback to another kernel to mistake for a
    pass: a single-entry list leaves torch nothing else enabled to pick.

    Yields a :class:`ForcedSdpaCalls` whose ``count`` a caller should assert is
    nonzero after the block. That is what distinguishes "the forced kernel ran"
    from "the patch never got looked at", the way a refactor that bound
    ``sdpa_kernel`` at import time inside the core would leave it.

    Scope this tightly around a single forward call: the backward triggered
    later replays the kernel autograd already recorded at that forward, so it
    needs no context of its own, and anything computed *outside* the block
    (e.g. a reference run on the ordinary, unforced priority list) must stay
    that way or the comparison stops meaning anything.
    """
    real_sdpa_kernel = torch.nn.attention.sdpa_kernel
    calls = ForcedSdpaCalls()

    def _forced(backends: Any, **kwargs: Any) -> Any:
        calls.record(backends)
        return real_sdpa_kernel([backend], **kwargs)

    torch.nn.attention.sdpa_kernel = _forced  # type: ignore[method-assign]
    try:
        yield calls
    finally:
        torch.nn.attention.sdpa_kernel = real_sdpa_kernel  # type: ignore[method-assign]


#: Placement marking a tensor as replicated (not sharded) across a mesh dimension.
REPLICATED = "replicated"

#: A placement is either the dimension a tensor is sharded along, or REPLICATED.
Placement = int | str


class ShardSpec:
    """
    Describe how a tensor is laid out over the named axes of a process mesh.

    Construct it with one keyword per sharded mesh axis, giving the tensor dimension
    that axis splits::

        ShardSpec(row=3, col=1)  # 2-D row/col mesh: dim 3 over row, dim 1 over col
        ShardSpec(sh=2, sw=3, col=1)  # N-way spatial mesh
        ShardSpec(
            row=REPLICATED, col=0
        )  # replicated over row, sharded on dim 0 over col

    Any axis not named is treated as ``REPLICATED``. The axis names must match the keys
    of the ``process_mesh_dict`` passed to :func:`check_parity` (``model`` excluded).
    """

    def __init__(self, **placements: Placement) -> None:
        self.placements: dict[str, Placement] = dict(placements)

    def placement(self, axis: str) -> Placement:
        """Return the placement for ``axis``, defaulting to ``REPLICATED``."""
        return self.placements.get(axis, REPLICATED)

    def sharded_axes(self) -> list[str]:
        """Axis names this spec shards a real tensor dimension along."""
        return [a for a, p in self.placements.items() if isinstance(p, int)]

    def __repr__(self) -> str:
        inner = ", ".join(f"{a}={p!r}" for a, p in self.placements.items())
        return f"ShardSpec({inner})"


#: A parameter is compared either through a ShardSpec or through a custom gather
#: function, for layouts that are permuted rather than simply sliced.
ParamRule = ShardSpec | Callable[[torch.Tensor], torch.Tensor]


def _mesh_axes(
    process_mesh_dict: Mapping[str, dist.ProcessGroup],
    mesh_axes: Sequence[str] | None,
    pm_key_model: str,
) -> list[str]:
    """Ordered list of sharded mesh axes (everything but the model group).

    The order matters: :func:`gather_tensor` decodes a model-group rank back into
    per-axis ranks assuming the mesh was laid out row-major over these axes, which is
    what ``init_device_mesh`` does for both ``make_process_mesh`` and
    ``make_multi_spatial_mesh`` in ``tests/process_mesh``.
    """
    if mesh_axes is not None:
        return list(mesh_axes)
    return [k for k in process_mesh_dict if k != pm_key_model]


def sync_device(device: torch.device) -> None:
    """
    Wait for every stream on the device, including the ones NCCL uses internally.

    jigsaw's collectives run on NCCL's own streams, so reading ``.grad`` straight after
    ``backward()`` returns can race with them. Call this before comparing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _all_gather_uneven(
    local: torch.Tensor, group: dist.ProcessGroup
) -> list[torch.Tensor]:
    """
    All-gather tensors that may differ in shape between ranks.

    ``dist.all_gather`` needs every tensor the same size, so the halo regime -- where
    ``kernel != stride`` leaves neighbouring ranks with output shards that differ by a
    row -- cannot use it directly. Gather the shapes first; if they already agree take
    the fast path, otherwise pad every rank up to the elementwise-max shape, gather, and
    narrow each part back to its real extent.
    """
    world = group.size()
    local = local.contiguous()
    shape = torch.tensor(local.shape, device=local.device, dtype=torch.long)
    shape_list = [torch.empty_like(shape) for _ in range(world)]
    dist.all_gather(shape_list, shape, group=group)
    shapes = [torch.Size(s.tolist()) for s in shape_list]

    if all(s == shapes[0] for s in shapes):
        parts = [torch.empty_like(local) for _ in range(world)]
        dist.all_gather(parts, local, group=group)
        return parts

    max_shape = [max(s[d] for s in shapes) for d in range(local.dim())]
    padded = local.new_zeros(max_shape)
    padded[tuple(slice(0, n) for n in local.shape)] = local
    gathered = [torch.empty_like(padded) for _ in range(world)]
    dist.all_gather(gathered, padded, group=group)
    return [
        g[tuple(slice(0, n) for n in shapes[i])].contiguous()
        for i, g in enumerate(gathered)
    ]


def split_tensor(
    x: torch.Tensor,
    spec: ShardSpec,
    process_mesh_dict: Mapping[str, dist.ProcessGroup],
    mesh_axes: Sequence[str] | None = None,
    pm_key_model: str = "model",
) -> torch.Tensor:
    """Take this rank's shard of a global tensor, following ``spec``."""
    axes = _mesh_axes(process_mesh_dict, mesh_axes, pm_key_model)
    out = x
    for axis in axes:
        placement = spec.placement(axis)
        if not isinstance(placement, int):
            continue
        group = process_mesh_dict[axis]
        size, rank = group.size(), group.rank()
        assert out.shape[placement] % size == 0, (
            f"dimension {placement} of size {out.shape[placement]} is not divisible "
            f"by the {axis!r} group size {size}"
        )
        out = out.tensor_split(size, dim=placement)[rank]
    return out.contiguous()


def _prod(values: Sequence[int]) -> int:
    out = 1
    for v in values:
        out *= v
    return out


def gather_tensor(
    local: torch.Tensor,
    spec: ShardSpec,
    process_mesh_dict: Mapping[str, dist.ProcessGroup],
    device: torch.device,
    name: str = "tensor",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    mesh_axes: Sequence[str] | None = None,
    pm_key_model: str = "model",
) -> torch.Tensor:
    """
    Reassemble a global tensor from the per-rank shards, following ``spec``.

    Sharded dimensions are concatenated. Replicated dimensions are checked for
    agreement across the group and then collapsed; a disagreement fails on every rank,
    since it means the replicas were never synchronized.
    """
    axes = _mesh_axes(process_mesh_dict, mesh_axes, pm_key_model)
    sizes = [process_mesh_dict[a].size() for a in axes]
    model_group = process_mesh_dict[pm_key_model]

    parts = _all_gather_uneven(local.contiguous(), model_group)
    assert len(parts) == _prod(sizes), (
        f"{name}: gathered {len(parts)} shards but the mesh axes {axes} multiply to "
        f"{_prod(sizes)}"
    )
    return _nd_merge(parts, axes, sizes, spec, name, atol, rtol, device, model_group)


def _nd_merge(
    parts: list[torch.Tensor],
    axes: Sequence[str],
    sizes: Sequence[int],
    spec: ShardSpec,
    name: str,
    atol: float,
    rtol: float,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Fold the flat list of shards back along one mesh axis at a time.

    ``parts`` is indexed by model-group rank, which -- with a row-major mesh -- is
    ``((r0 * n1) + r1) * n2 + ...``. The first axis therefore varies slowest, so its
    ``n0`` blocks are contiguous runs of ``len(parts) // n0``.
    """
    if not sizes:
        return parts[0]

    n, axis = sizes[0], axes[0]
    block = len(parts) // n
    merged = [
        _nd_merge(
            list(parts[i * block : (i + 1) * block]),
            axes[1:],
            sizes[1:],
            spec,
            name,
            atol,
            rtol,
            device,
            group,
        )
        for i in range(n)
    ]
    return _merge(
        merged,
        spec.placement(axis),
        f"{name} across {axis!r}",
        atol,
        rtol,
        device,
        group,
    )


def _merge(
    parts: list[torch.Tensor],
    placement: Placement,
    what: str,
    atol: float,
    rtol: float,
    device: torch.device,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """Concatenate shards, or collapse replicas after checking that they agree."""
    if isinstance(placement, int):
        return torch.cat(parts, dim=placement)

    ok = True
    worst = 0.0
    for part in parts[1:]:
        if not torch.allclose(parts[0], part, atol=atol, rtol=rtol):
            ok = False
            worst = max(worst, (parts[0] - part).abs().max().item())
    if not all_ranks_ok(ok, device, group):
        raise AssertionError(
            f"{what}: replicas disagree (max abs diff {worst:.3e}). The tensor is "
            f"declared REPLICATED, so the synchronizing all-reduce never ran or did "
            f"not complete."
        )
    return parts[0]


def all_ranks_ok(ok: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    """
    Reduce a per-rank check with logical AND so every rank reaches the same verdict.

    Without this a failing rank raises while the others block in the next collective,
    which hangs the job instead of failing it.
    """
    if not dist.is_initialized() or group.size() == 1:
        return ok
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def assert_close(
    name: str,
    got: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
    device: torch.device,
    group: dist.ProcessGroup,
) -> None:
    """Compare two tensors, failing on every rank if any rank disagrees.

    Use this instead of a bare ``assert torch.allclose(...)`` for any comparison that
    sits after a collective (a gather, a grad-sync all-reduce): a bare assert that trips
    on a single rank leaves the others waiting in the next collective and the job hangs
    rather than failing.
    """
    shape_ok = got.shape == expected.shape
    if not all_ranks_ok(shape_ok, device, group):
        raise AssertionError(
            f"{name}: gathered shape {tuple(got.shape)} != reference shape "
            f"{tuple(expected.shape)}"
        )
    ok = torch.allclose(got, expected, atol=atol, rtol=rtol)
    diff = (got - expected).abs().max().item()
    if not all_ranks_ok(ok, device, group):
        raise AssertionError(
            f"{name} does not match the serial reference "
            f"(max abs diff {diff:.3e}, atol={atol}, rtol={rtol})"
        )


def check_parity(*args: Any, **kwargs: Any) -> None:
    """Compare a distributed module against its serial counterpart, with TF32 off.

    Thin wrapper around :func:`_check_parity`; see that function for the parameters.
    """
    with no_tf32():
        _check_parity(*args, **kwargs)


def _check_parity(
    serial_module: torch.nn.Module,
    build_distributed: Callable[[torch.nn.Module], torch.nn.Module],
    input_shape: tuple[int, ...],
    input_spec: ShardSpec,
    output_spec: ShardSpec,
    param_rules: dict[str, ParamRule],
    process_mesh_dict: Mapping[str, dist.ProcessGroup],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    grad_atol: float | None = None,
    grad_rtol: float | None = None,
    seed: int = 333,
    check_param_values: bool = True,
    mesh_axes: Sequence[str] | None = None,
    pm_key_model: str = "model",
    serial_forward: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    | None = None,
) -> None:
    """
    Compare a distributed module against its serial counterpart, forward and backward.

    The caller seeds and builds ``serial_module`` (so every rank builds an identical
    reference), then hands over ``build_distributed`` to turn it into the sharded layer.

    serial_module: torch.nn.Module
        The serial reference. Its forward and backward run *before* ``build_distributed``
        is called, so an in-place converter cannot disturb the reference values.
    build_distributed: Callable
        Produces the distributed module from the serial one, and must return a module
        whose parameters already match the serial ones. Either a converter
        (``jigsaw.convert.convert_to_jigsaw``) or a direct ``Distributed*(...)``
        construction followed by copying the appropriate serial-weight shards in.
    input_shape: tuple[int, ...]
        Shape of the *global* input.
    input_spec, output_spec: ShardSpec
        Layout of the module's input and output over the mesh axes.
    param_rules: dict[str, ParamRule]
        One entry per parameter of the distributed module, keyed by the name it has in
        both modules. Every parameter must appear, so that adding one to a layer without
        extending the test fails loudly instead of going unchecked. A value is either a
        ``ShardSpec`` (gathered like a tensor) or a callable taking the local parameter
        and returning the reassembled global one (for permuted layouts).
    grad_atol, grad_rtol: float | None
        Tolerances for gradients; default to ``atol`` / ``rtol``. Gradients accumulate
        over the batch and over every sharded position, so they usually need more slack.
    check_param_values: bool
        Also compare parameter *values*, not just gradients. Turn off for layers whose
        conversion deliberately permutes a parameter into a different layout.
    mesh_axes: Sequence[str] | None
        Order of the sharded mesh axes; defaults to every ``process_mesh_dict`` key
        except ``model``, which is what the ``process_mesh`` fixtures produce.
    serial_forward: Callable | None
        How to evaluate the reference, defaulting to ``serial_module(x)``. Override when
        the distributed layer is not equivalent to a plain serial call over the whole
        input (e.g. non-ring attention, where each spatial rank attends only within its
        own token shard and the reference has to window the sequence the same way).
    """
    grad_atol = atol if grad_atol is None else grad_atol
    grad_rtol = rtol if grad_rtol is None else grad_rtol
    model_group = process_mesh_dict[pm_key_model]
    axes = _mesh_axes(process_mesh_dict, mesh_axes, pm_key_model)

    # Every rank draws the same global tensors.
    torch.manual_seed(seed)
    x_global = torch.randn(input_shape, dtype=dtype, device=device, requires_grad=True)

    y_ref = (
        serial_module(x_global)
        if serial_forward is None
        else serial_forward(serial_module, x_global)
    )
    g_global = torch.randn_like(y_ref)
    y_ref.backward(g_global)

    assert x_global.grad is not None, "serial backward produced no input gradient"
    ref_output = y_ref.detach().clone()
    ref_dx = x_global.grad.detach().clone()
    ref_params = {n: p.detach().clone() for n, p in serial_module.named_parameters()}
    ref_grads = {
        n: (None if p.grad is None else p.grad.detach().clone())
        for n, p in serial_module.named_parameters()
    }

    # Conversion happens only now: a converter may mutate the serial module.
    dist_module = build_distributed(serial_module)

    x_local = (
        split_tensor(
            x_global.detach(), input_spec, process_mesh_dict, axes, pm_key_model
        )
        .clone()
        .requires_grad_(True)
    )
    g_local = split_tensor(
        g_global, output_spec, process_mesh_dict, axes, pm_key_model
    ).clone()

    y_local = dist_module(x_local)
    y_local.backward(g_local)

    # Drain the gradient collectives before reading .grad.
    sync_device(device)

    assert x_local.grad is not None, "distributed backward produced no input gradient"
    assert_close(
        "output",
        gather_tensor(
            y_local,
            output_spec,
            process_mesh_dict,
            device,
            "output",
            atol,
            rtol,
            axes,
            pm_key_model,
        ),
        ref_output,
        atol,
        rtol,
        device,
        model_group,
    )
    assert_close(
        "input gradient",
        gather_tensor(
            x_local.grad,
            input_spec,
            process_mesh_dict,
            device,
            "input gradient",
            grad_atol,
            grad_rtol,
            axes,
            pm_key_model,
        ),
        ref_dx,
        grad_atol,
        grad_rtol,
        device,
        model_group,
    )

    dist_params = dict(dist_module.named_parameters())
    unspecified = sorted(set(dist_params) - set(param_rules))
    assert not unspecified, (
        f"no parity rule given for distributed parameter(s): {unspecified}. "
        f"Every parameter must be listed so none is silently left unchecked."
    )

    for pname, rule in param_rules.items():
        param = dist_params.get(pname)
        assert param is not None, f"distributed module has no parameter {pname!r}"

        def _gather(tensor: torch.Tensor, label: str) -> torch.Tensor:
            if isinstance(rule, ShardSpec):
                return gather_tensor(
                    tensor,
                    rule,
                    process_mesh_dict,
                    device,
                    label,
                    grad_atol,
                    grad_rtol,
                    axes,
                    pm_key_model,
                )
            return rule(tensor)

        if check_param_values and pname in ref_params:
            assert_close(
                f"{pname} value",
                _gather(param.detach(), f"{pname} value"),
                ref_params[pname],
                atol,
                rtol,
                device,
                model_group,
            )

        ref_grad = ref_grads.get(pname)
        if ref_grad is None:
            continue
        assert param.grad is not None, (
            f"{pname}: serial reference has a gradient but the distributed parameter "
            f"does not"
        )
        assert_close(
            f"{pname} gradient",
            _gather(param.grad, f"{pname} gradient"),
            ref_grad,
            grad_atol,
            grad_rtol,
            device,
            model_group,
        )
