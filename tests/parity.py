"""
Shared harness for comparing a distributed jigsaw layer against a serial reference.

The pattern implemented here mirrors the distributed tests in torch-harmonics: build
the serial module, run a forward and backward pass on the *global* tensors, then split
the input and the upstream gradient across the process mesh, run the distributed
module, gather its results back and compare.

Two jigsaw-specific points shape the design:

* Backward is driven with an explicit upstream gradient rather than a loss. jigsaw's
  weight-gradient all-reduce (`jigsaw.hooks.make_grad_sync_hook`) is a `ReduceOp.SUM`
  over the spatial group, so the distributed gradient only equals the serial one when
  nothing averages over a sharded dimension. Splitting a global `randn` upstream
  gradient makes that relation exact and removes any need for a scaling factor.

* Parameters are described as either *sharded* over a group (compare by gathering) or
  *replicated* over it (compare by checking the replicas agree, then taking one). For a
  replicated parameter the agreement check is itself the test that the grad-sync hook
  fired: without the all-reduce the per-rank gradients hold only local contributions
  and differ.
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist


@contextlib.contextmanager
def no_tf32() -> Iterator[None]:
    """
    Disable TF32 for the duration of a parity comparison, restoring it afterwards.

    torch.backends.cudnn.allow_tf32 defaults to True, which gives fp32 convolutions a
    10-bit mantissa and roughly 1e-3 relative error. That alone would be tolerable if
    both sides made the same error, but the serial and distributed paths contract the
    channel axis in a different order and pick different cuDNN algorithms, so the two
    errors do not cancel. The drift grows with the reduction length (in_channels *
    kernel * kernel), which makes the tolerance a layer-shape property rather than a
    property of the distribution being tested.

    Disabling it keeps the comparison about the sharding, which is what these tests are
    for. torch-harmonics does the same in its distributed convolution tests.
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

#: Placement marking a tensor as replicated (not sharded) across a mesh dimension.
REPLICATED = "replicated"

#: A placement is either the dimension a tensor is sharded along, or REPLICATED.
Placement = int | str


@dataclass(frozen=True)
class ShardSpec:
    """
    Describe how a tensor is laid out over the two mesh dimensions.

    row: int | str
        Dimension sharded across the row (spatial / sequence) group, or REPLICATED.
    col: int | str
        Dimension sharded across the column (channel) group, or REPLICATED.
    """

    row: Placement = REPLICATED
    col: Placement = REPLICATED


def sync_device(device: torch.device) -> None:
    """
    Wait for every stream on the device, including the ones NCCL uses internally.

    jigsaw's grad-sync hooks issue `all_reduce(..., async_op=True)` and only wait on the
    handles from an autograd engine callback, so reading `.grad` straight after
    `backward()` returns can race with the collective. Call this before comparing.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def split_tensor(
    x: torch.Tensor,
    spec: ShardSpec,
    process_mesh_dict: dict[str, dist.ProcessGroup],
    pm_key_row: str = "row",
    pm_key_col: str = "col",
) -> torch.Tensor:
    """Take this rank's shard of a global tensor, following `spec`."""
    out = x
    for placement, key in ((spec.col, pm_key_col), (spec.row, pm_key_row)):
        if not isinstance(placement, int):
            continue
        group = process_mesh_dict[key]
        size, rank = group.size(), group.rank()
        assert out.shape[placement] % size == 0, (
            f"dimension {placement} of size {out.shape[placement]} is not divisible "
            f"by the {key} group size {size}"
        )
        out = out.tensor_split(size, dim=placement)[rank]
    return out.contiguous()


def gather_tensor(
    local: torch.Tensor,
    spec: ShardSpec,
    process_mesh_dict: dict[str, dist.ProcessGroup],
    name: str = "tensor",
    atol: float = 1e-5,
    rtol: float = 1e-5,
    pm_key_row: str = "row",
    pm_key_col: str = "col",
    pm_key_model: str = "model",
) -> torch.Tensor:
    """
    Reassemble a global tensor from the per-rank shards, following `spec`.

    Sharded dimensions are concatenated. Replicated dimensions are checked for
    agreement across the group and then collapsed; a disagreement raises, since it
    means the replicas were never synchronized.
    """
    row_group, col_group = process_mesh_dict[pm_key_row], process_mesh_dict[pm_key_col]
    n_rows, n_cols = row_group.size(), col_group.size()

    local = local.contiguous()
    parts = [torch.empty_like(local) for _ in range(n_rows * n_cols)]
    dist.all_gather(parts, local, group=process_mesh_dict[pm_key_model])

    # The model group is laid out row-major: model rank == row_rank * n_cols + col_rank.
    rows = [
        _merge(
            parts[i * n_cols : (i + 1) * n_cols],
            spec.col,
            f"{name} across the column group",
            atol,
            rtol,
        )
        for i in range(n_rows)
    ]
    return _merge(rows, spec.row, f"{name} across the row group", atol, rtol)


def _merge(
    parts: list[torch.Tensor],
    placement: Placement,
    what: str,
    atol: float,
    rtol: float,
) -> torch.Tensor:
    """Concatenate shards, or collapse replicas after checking that they agree."""
    if isinstance(placement, int):
        return torch.cat(parts, dim=placement)

    for i, part in enumerate(parts[1:], start=1):
        if not torch.allclose(parts[0], part, atol=atol, rtol=rtol):
            diff = (parts[0] - part).abs().max().item()
            raise AssertionError(
                f"{what}: replica {i} disagrees with replica 0 "
                f"(max abs diff {diff:.3e}). The tensor is declared REPLICATED, so "
                f"this means the synchronizing all-reduce never ran or did not complete."
            )
    return parts[0]


def all_ranks_ok(ok: bool, device: torch.device, group: dist.ProcessGroup) -> bool:
    """
    Reduce a per-rank check with logical AND so every rank reaches the same verdict.

    Without this a failing rank raises while the others block in the next collective,
    which hangs the job instead of failing it. Borrowed from torch-harmonics'
    `reduce_success`.
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
    """Compare two tensors, failing on every rank if any rank disagrees."""
    assert got.shape == expected.shape, (
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


#: A parameter is compared either through a ShardSpec or through a custom gather
#: function, for layouts that are permuted rather than simply sliced.
ParamRule = ShardSpec | Callable[[torch.Tensor], torch.Tensor]


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
    process_mesh_dict: dict[str, dist.ProcessGroup],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    grad_atol: float | None = None,
    grad_rtol: float | None = None,
    seed: int = 333,
    check_param_values: bool = True,
    serial_forward: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    | None = None,
) -> None:
    """
    Compare a distributed module against its serial counterpart, forward and backward.

    The caller is responsible for seeding before constructing `serial_module`, so that
    every rank builds an identical reference.

    serial_module: torch.nn.Module
        The serial reference. Its forward and backward run before `build_distributed`
        is called, so an in-place converter cannot disturb the reference values.
    build_distributed: Callable
        Produces the distributed module from the serial one, e.g. a lambda around
        `jigsaw.convert.convert_to_jigsaw`.
    input_shape: tuple[int, ...]
        Shape of the *global* input.
    input_spec, output_spec: ShardSpec
        Layout of the module's input and output over the mesh.
    param_rules: dict[str, ParamRule]
        One entry per parameter of the distributed module, keyed by the name it has in
        both modules. Every parameter must appear, so that adding one to a layer
        without extending the test fails loudly instead of going unchecked.
    grad_atol, grad_rtol: float | None
        Tolerances for gradients; default to `atol` / `rtol`. Gradients accumulate over
        the batch and over every sharded position, so they usually need more slack than
        the forward pass.
    check_param_values: bool
        Also compare parameter *values* (not just gradients). Turn off for layers whose
        conversion deliberately permutes a parameter into a different layout.
    serial_forward: Callable | None
        How to evaluate the reference, defaulting to `serial_module(x)`. Override it
        when the distributed layer is not equivalent to a plain serial call over the
        whole input, for example attention without ring communication, where each row
        rank attends only within its own token shard and the reference has to window
        the sequence the same way.
    """
    grad_atol = atol if grad_atol is None else grad_atol
    grad_rtol = rtol if grad_rtol is None else grad_rtol
    model_group = process_mesh_dict["model"]

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

    # Conversion happens only now: `convert_to_jigsaw` may mutate the serial module.
    dist_module = build_distributed(serial_module)

    x_local = (
        split_tensor(x_global.detach(), input_spec, process_mesh_dict)
        .clone()
        .requires_grad_(True)
    )
    g_local = split_tensor(g_global, output_spec, process_mesh_dict).clone()

    y_local = dist_module(x_local)
    y_local.backward(g_local)

    # The grad-sync hooks all-reduce asynchronously; drain before reading .grad.
    sync_device(device)

    assert x_local.grad is not None, "distributed backward produced no input gradient"
    assert_close(
        "output",
        gather_tensor(y_local, output_spec, process_mesh_dict, "output", atol, rtol),
        ref_output,
        atol,
        rtol,
        device,
        model_group,
    )
    assert_close(
        "input gradient",
        gather_tensor(
            x_local.grad, input_spec, process_mesh_dict, "input gradient", atol, rtol
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

    for name, rule in param_rules.items():
        param = dist_params.get(name)
        assert param is not None, f"distributed module has no parameter {name!r}"

        def _gather(tensor: torch.Tensor, label: str) -> torch.Tensor:
            if isinstance(rule, ShardSpec):
                return gather_tensor(
                    tensor, rule, process_mesh_dict, label, grad_atol, grad_rtol
                )
            return rule(tensor)

        if check_param_values and name in ref_params:
            assert_close(
                f"{name} value",
                _gather(param.detach(), f"{name} value"),
                ref_params[name],
                atol,
                rtol,
                device,
                model_group,
            )

        ref_grad = ref_grads.get(name)
        if ref_grad is None:
            continue
        assert param.grad is not None, (
            f"{name}: serial reference has a gradient but the distributed parameter "
            f"does not"
        )
        assert_close(
            f"{name} gradient",
            _gather(param.grad, f"{name} gradient"),
            ref_grad,
            grad_atol,
            grad_rtol,
            device,
            model_group,
        )
