"""
Parity tests for DistributedConv2D against a serial torch.nn.Conv2d reference.

Run under a multi-rank launcher, see tests/run_distributed_tests.sh. The mesh shape
comes from the GRID_ROW / GRID_COL environment variables.

Conv is far more restrictive about the grid than the other layers:

* The channel group drives an all-to-all whose implementation is selected by size in
  ConvConfig._channel_comm_fn_selection, which only handles 1 and 2 and raises
  NotImplementedError otherwise. GRID_COL is therefore limited to 1 or 2.
* The spatial group drives the halo exchange, and
  ConvConfig._communicate_halos_fn_selection only assigns the halo helpers for group
  sizes 2 and 4. Any other size leaves them unset. GRID_ROW is therefore 2 or 4.

So on four ranks the only grid conv can run on is 2 x 2, and on eight ranks 4 x 2.
Every other shape skips.

Only kernel_size == stride is covered here. That is the configuration in which
ConvConfig._communicate_halos_selection returns the no-op halo pair, so the shards are
independent and every rank produces an output of the same size. With kernel != stride
the halo exchange runs and the per-rank output widths differ, which the gather in
tests/parity.py cannot express yet -- it allocates the all-gather buffers with
torch.empty_like and so assumes uniform shards.
"""

import pytest
import torch

import jigsaw.convert as convert
from jigsaw.modules.conv import DistributedConv2D
from tests.parity import REPLICATED, ShardSpec, check_parity
from tests.process_mesh import *  # noqa: F403

pytestmark = [pytest.mark.gpu_required, pytest.mark.dist_required]

# Sizes chosen so that kernel == stride and the width divides evenly by both spatial
# group sizes conv supports (2 and 4), keeping the shards uniform. Rectangular on
# purpose, so a height/width mix-up cannot pass unnoticed.
_CASES = [
    # (batch, in_ch, out_ch, height, width, kernel, atol, rtol, gatol, grtol)
    (2, 4, 8, 32, 64, 4, 1e-5, 1e-5, 1e-4, 1e-4),
    (2, 8, 8, 18, 36, 2, 1e-5, 1e-5, 1e-4, 1e-4),
    (1, 4, 16, 32, 64, 8, 1e-5, 1e-5, 1e-4, 1e-4),
]


def _skip_unless_grid_supported(process_mesh_dict: dict, width: int, kernel: int) -> None:
    """
    Skip identically on every rank when conv cannot run on this grid.

    Every condition is derived from the mesh shape and the case parameters, which are
    identical on all ranks, so no rank can skip while another proceeds into a collective.
    """
    row_size = process_mesh_dict["row"].size()
    col_size = process_mesh_dict["col"].size()

    if col_size not in (1, 2):
        pytest.skip(
            f"conv supports a channel group of 1 or 2, not {col_size} "
            f"(ConvConfig._channel_comm_fn_selection)"
        )
    if row_size not in (2, 4):
        pytest.skip(
            f"conv supports a spatial group of 2 or 4, not {row_size} "
            f"(ConvConfig._communicate_halos_fn_selection)"
        )
    if width % (kernel * row_size) != 0:
        pytest.skip(
            f"width {width} does not split evenly into {row_size} shards of whole "
            f"kernels of size {kernel}"
        )


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "batch,in_ch,out_ch,height,width,kernel,atol,rtol,grad_atol,grad_rtol", _CASES
)
def test_distributed_conv2d_parity(
    device: torch.device,
    process_mesh_dict_env: dict,
    bias: bool,
    batch: int,
    in_ch: int,
    out_ch: int,
    height: int,
    width: int,
    kernel: int,
    atol: float,
    rtol: float,
    grad_atol: float,
    grad_rtol: float,
) -> None:
    """Compare forward, input gradient and parameter gradients against serial conv2d."""
    _skip_unless_grid_supported(process_mesh_dict_env, width, kernel)

    col_size = process_mesh_dict_env["col"].size()
    if in_ch % col_size or out_ch % col_size:
        pytest.skip(
            f"channels ({in_ch}, {out_ch}) do not divide by the channel group size "
            f"{col_size}"
        )

    # All ranks must build an identical serial reference.
    torch.manual_seed(1)
    serial = torch.nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        stride=kernel,
        padding=0,
        bias=bias,
        device=device,
    )

    def build_distributed(module: torch.nn.Module) -> torch.nn.Module:
        converted = convert.convert_to_jigsaw(module, process_mesh_dict_env, device)
        assert isinstance(converted, DistributedConv2D)
        return converted

    # Data is [N, C, H, W]: channels split over the column group, width over the row
    # group. Height stays whole -- the halo exchange in ConvConfig operates on the last
    # dimension only.
    data_spec = ShardSpec(row=3, col=1)

    # Both parameters are sharded over the channel group along their output-channel
    # dimension and replicated across the spatial group, where conv.py registers
    # make_grad_sync_hook(spatial_group) on each. Declaring them REPLICATED makes the
    # harness assert the spatial replicas agree, which is what verifies those hooks fired.
    param_rules: dict = {"weight": ShardSpec(row=REPLICATED, col=0)}
    if bias:
        param_rules["bias"] = ShardSpec(row=REPLICATED, col=0)

    check_parity(
        serial_module=serial,
        build_distributed=build_distributed,
        input_shape=(batch, in_ch, height, width),
        input_spec=data_spec,
        output_spec=data_spec,
        param_rules=param_rules,
        process_mesh_dict=process_mesh_dict_env,
        device=device,
        atol=atol,
        rtol=rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
    )
