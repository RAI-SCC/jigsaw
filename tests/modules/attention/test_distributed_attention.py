"""
Parity tests for DistributedAttention against the serial Attention reference.

Run under a multi-rank launcher, see tests/run_distributed_tests.sh. The mesh shape
comes from the GRID_ROW / GRID_COL environment variables.

Sequence lengths are written as a rectangular spatial grid (lat x lon) flattened into
the token dimension, which is how attention sees the data in the models jigsaw targets.
"""

import os

import pytest
import torch

import jigsaw.convert as convert
from jigsaw.modules.attention import Attention, DistributedAttention
from jigsaw.utils_aggregate import gather_1d_results_list
from tests.parity import REPLICATED, ShardSpec, check_parity
from tests.process_mesh import *  # noqa: F403

pytestmark = [pytest.mark.gpu_required, pytest.mark.dist_required]

# The 90 x 180 case is a 16200-token sequence; the serial reference attention matrix
# alone is 16200^2 per head, so it is opt-in rather than part of the routine run.
_run_slow_tests = os.getenv("JIGSAW_RUN_SLOW_TESTS", "0") == "1"
_SLOW_SHAPES = frozenset({(90, 180)})

# q/k/v are cast to bfloat16 inside both Attention and DistributedAttention, so even
# the "fp32" cases carry bf16 attention error. Tolerances are per-case rather than
# global: gradients accumulate over the batch and over every sharded token, so they
# need considerably more slack than the forward pass.
_CASES = [
    # (nlat, nlon, batch, in_dim, embed_dim, out_dim, heads, atol, rtol, gatol, grtol)
    (12, 20, 2, 96, 96, 96, 4, 2e-2, 2e-2, 1e-1, 5e-2),
    (18, 36, 2, 96, 96, 96, 4, 2e-2, 2e-2, 1e-1, 5e-2),
    (12, 20, 2, 96, 192, 96, 8, 2e-2, 2e-2, 1e-1, 5e-2),
    (18, 36, 1, 48, 96, 48, 4, 2e-2, 2e-2, 1e-1, 5e-2),
    (90, 180, 1, 96, 96, 96, 4, 3e-2, 3e-2, 3e-1, 5e-2),
]


def _skip_unless_shapes_fit(
    process_mesh_dict: dict,
    seq: int,
    in_dim: int,
    embed_dim: int,
    out_dim: int,
    heads: int,
) -> None:
    """
    Skip identically on every rank when a case does not fit the mesh.

    The skip decision is derived only from the mesh shape and the case parameters,
    both of which are the same on all ranks, so no rank can skip while another runs
    on into a collective.
    """
    row_size = process_mesh_dict["row"].size()
    col_size = process_mesh_dict["col"].size()

    requirements = {
        "sequence length": (seq, row_size),
        "input dim over the column group": (in_dim, col_size),
        "input dim over the row group": (in_dim, row_size),
        "embed dim over the row group": (embed_dim, row_size),
        "embed dim over the column group": (embed_dim, col_size),
        "qkv output dim": (3 * embed_dim, col_size),
        "output dim": (out_dim, col_size),
        "heads": (heads, col_size),
    }
    for what, (value, size) in requirements.items():
        if value % size != 0:
            pytest.skip(f"{what}: {value} is not divisible by the group size {size}")


def _make_qkv_gather(
    process_mesh_dict: dict, heads: int, embed_dim: int, in_dim: int
):
    """
    Build the inverse of the head permutation applied to the fused qkv weight.

    `Attention.convert_to_jigsaw` does not simply slice the qkv weight. It re-views it
    as [3, heads, head_dim, in_dim], splits the head axis across the column group, and
    writes head group `i` into row block `i` of the local weight, keeping only the
    input columns belonging to this rank. Gathering the shards therefore yields a
    permuted tensor, and comparing it against the serial weight requires undoing that
    mapping first.

    The returned function reassembles the global [3 * embed_dim, in_dim] weight, and
    works for gradients as well since they carry the same layout. It gathers over the
    column group only, matching the converter's own assumption that the row group is 1;
    row groups larger than 1 are xfailed in the test below for that same reason.
    """
    col_size = process_mesh_dict["col"].size()
    head_dim = embed_dim // heads
    heads_per_shard = heads // col_size
    in_shard = in_dim // col_size
    row_block = 3 * embed_dim // col_size

    def gather(local: torch.Tensor) -> torch.Tensor:
        parts = gather_1d_results_list(local, process_mesh_dict, pm_key="col")
        out = torch.empty(
            3, heads, head_dim, in_dim, dtype=local.dtype, device=local.device
        )
        for col_rank, part in enumerate(parts):
            for i in range(col_size):
                block = part[i * row_block : (i + 1) * row_block].reshape(
                    3, heads_per_shard, head_dim, in_shard
                )
                out[
                    :,
                    i * heads_per_shard : (i + 1) * heads_per_shard,
                    :,
                    col_rank * in_shard : (col_rank + 1) * in_shard,
                ] = block
        return out.reshape(3 * embed_dim, in_dim)

    return gather


@pytest.mark.parametrize("ring", [False, True])
@pytest.mark.parametrize(
    "nlat,nlon,batch,in_dim,embed_dim,out_dim,heads,atol,rtol,grad_atol,grad_rtol",
    _CASES,
)
def test_distributed_attention_parity(
    device: torch.device,
    process_mesh_dict_env: dict,
    ring: bool,
    nlat: int,
    nlon: int,
    batch: int,
    in_dim: int,
    embed_dim: int,
    out_dim: int,
    heads: int,
    atol: float,
    rtol: float,
    grad_atol: float,
    grad_rtol: float,
) -> None:
    """Compare forward, input gradient and parameter gradients against serial attention."""
    if (nlat, nlon) in _SLOW_SHAPES and not _run_slow_tests:
        pytest.skip("slow case; set JIGSAW_RUN_SLOW_TESTS=1 to run")

    seq = nlat * nlon
    _skip_unless_shapes_fit(
        process_mesh_dict_env, seq, in_dim, embed_dim, out_dim, heads
    )

    row_size = process_mesh_dict_env["row"].size()

    if ring and row_size == 1:
        # DistributedAttention.__init__ asserts spatial_group.size() > 1 for ring
        # attention, so there is nothing to compare here.
        pytest.skip("ring attention requires a row group larger than 1")

    # The two paths are equivalent to different serial computations, and comparing
    # against the wrong one would fail on a correct implementation:
    #
    # ring=True   ring communication carries keys and values around the row group, so
    #             every token attends over the whole sequence. The reference is a plain
    #             serial forward.
    # ring=False  each row rank runs attention over its own token shard only, i.e.
    #             windowed attention. The reference has to window the sequence the same
    #             way. Attention is independent across the batch, so folding the windows
    #             into the batch dimension reproduces it exactly -- and it makes the
    #             serial weight gradients accumulate over the windows just as the
    #             spatial grad-sync hook sums them over the row group.
    def serial_forward(module: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        if ring or row_size == 1:
            return module(x)
        windowed = x.reshape(x.shape[0], row_size, seq // row_size, x.shape[2])
        out = module(windowed.flatten(0, 1))
        return out.unflatten(0, (x.shape[0], row_size)).flatten(1, 2)

    # All ranks must build an identical serial reference.
    torch.manual_seed(1)
    serial = Attention(
        in_dim=in_dim,
        embed_dim=embed_dim,
        out_dim=out_dim,
        heads=heads,
        device=device,
        dropout_p=0.0,
        qk_norm=False,
        dtype=torch.float32,
    )

    def build_distributed(module: torch.nn.Module) -> torch.nn.Module:
        converted = convert.convert_to_jigsaw(
            module, process_mesh_dict_env, device, ring=ring
        )
        assert isinstance(converted, DistributedAttention)
        return converted

    # Tokens are split over the row group, channels over the column group.
    token_channel_spec = ShardSpec(row=1, col=2)

    check_parity(
        serial_module=serial,
        build_distributed=build_distributed,
        input_shape=(batch, seq, in_dim),
        input_spec=token_channel_spec,
        output_spec=token_channel_spec,
        param_rules={
            # Both sublayers are built with model_kw=col_kw, so out_features stays
            # unsharded and the weight is replicated across the row group while its
            # in_features are split over the column group. Declaring it REPLICATED makes
            # the harness assert that the row replicas agree, which is what verifies the
            # spatial grad-sync hook actually fired.
            "head_concat.weight": ShardSpec(row=REPLICATED, col=1),
            # The fused qkv weight is permuted, not sliced; undo that before comparing.
            "qkv.weight": _make_qkv_gather(
                process_mesh_dict_env, heads, embed_dim, in_dim
            ),
        },
        process_mesh_dict=process_mesh_dict_env,
        device=device,
        atol=atol,
        rtol=rtol,
        grad_atol=grad_atol,
        grad_rtol=grad_rtol,
        serial_forward=serial_forward,
    )
