"""
Converter-path parity for ``DistributedConv{1,2,3}D``.

The rest of the convolution suite builds the distributed layer directly and rebuilds
the serial reference by gathering its shards. This file goes the other way: start from
a plain ``torch.nn.Conv{1,2,3}d``, run it, then hand it to
``jigsaw.convert.convert_to_jigsaw`` and check the sharded layer reproduces the serial
output, input gradient and -- the part the hand-rolled checks skip -- every parameter
value and parameter gradient, via the shared harness in ``tests/parity``.

Weight and bias are declared ``REPLICATED`` over the spatial (``row``) group: the
harness then asserts the replicas agree after backward, which is the test that
``conv.base`` registered its spatial grad-sync hook and the all-reduce completed.

All cases run on a 2 (spatial) x 2 (channel) mesh, so they need exactly 4 ranks.
"""

import pytest
import torch

import jigsaw.convert as convert
import jigsaw.modules.convolution as convolution
from tests.parity import REPLICATED, ShardSpec, _all_gather_uneven, check_parity

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]

_CONV_CLS = {
    1: convolution.DistributedConv1d,
    2: convolution.DistributedConv2d,
    3: convolution.DistributedConv3d,
}
_TORCH_CLS = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

# (n_spatial, batch, in_ch, out_ch, spatial_shape, kernel, stride, padding)
#   * kernel == stride, padding 0  -> no-halo tiling, uniform shards
#   * kernel != stride             -> halo exchange, exercises the ragged gather
_CASES = [
    (1, 2, 4, 8, (32,), 4, 4, 0),
    (2, 2, 4, 8, (16, 32), 4, 4, 0),
    (2, 2, 8, 8, (16, 32), 3, 2, 1),
    (3, 2, 4, 8, (8, 8, 16), 2, 2, 0),
]
_IDS = [f"conv{n}d_k{k}_s{s}_p{p}" for (n, _, _, _, _, k, s, p) in _CASES]


class TestConvParity:
    # ``groups`` is 1 (full-channel-gather path) or 2 == col.size() (every
    # channel rank owns whole conv-groups, no channel comm). The converter maps a
    # grouped ``torch.nn.Conv*`` the same way -- weight sharded on the output
    # axis over ``col``, replicated over ``row`` -- so the parity rules are
    # unchanged.
    @pytest.mark.parametrize("groups", [1, 2])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize(
        "n_spatial,batch,in_ch,out_ch,spatial,kernel,stride,padding", _CASES, ids=_IDS
    )
    def test_conv_converter_parity(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        groups: int,
        bias: bool,
        n_spatial: int,
        batch: int,
        in_ch: int,
        out_ch: int,
        spatial: tuple[int, ...],
        kernel: int,
        stride: int,
        padding: int,
    ) -> None:
        """Serial conv -> convert_to_jigsaw -> compare forward, input grad, param grads."""
        mesh = process_mesh_dict_2x2
        row_size = mesh["row"].size()

        # The width axis (last spatial dim) is the one the row group splits and the halo
        # exchange operates on; its global output must divide by the row group size.
        ext = kernel
        w_out = (spatial[-1] + 2 * padding - ext) // stride + 1
        if w_out % row_size != 0:
            pytest.skip(
                f"global output width {w_out} does not divide by the row group size "
                f"{row_size}"
            )

        # channel (dim 1) split over col, width (last dim) split over row.
        last = 1 + n_spatial
        data_spec = ShardSpec(col=1, row=last)
        param_rules: dict = {"weight": ShardSpec(col=0, row=REPLICATED)}
        if bias:
            param_rules["bias"] = ShardSpec(col=0, row=REPLICATED)

        torch.manual_seed(0)  # identical serial reference on every rank
        serial = _TORCH_CLS[n_spatial](
            in_ch,
            out_ch,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
            device=device,
        )

        def build_distributed(module: torch.nn.Module) -> torch.nn.Module:
            converted = convert.convert_to_jigsaw(module, mesh, device)
            assert isinstance(converted, _CONV_CLS[n_spatial])
            return converted

        check_parity(
            serial_module=serial,
            build_distributed=build_distributed,
            input_shape=(batch, in_ch, *spatial),
            input_spec=data_spec,
            output_spec=data_spec,
            param_rules=param_rules,
            process_mesh_dict=mesh,
            device=device,
            atol=1e-4,
            rtol=1e-4,
            grad_atol=1e-4,
            grad_rtol=1e-4,
        )

    def test_all_gather_uneven_ragged(self, device: torch.device) -> None:
        """``_all_gather_uneven`` reassembles shards whose shape differs between ranks."""
        world = torch.distributed.group.WORLD
        size = torch.distributed.get_world_size(world)
        rank = torch.distributed.get_rank(world)

        # rank r contributes a (2, r + 2) block, all filled with r; concatenating along
        # dim 1 must recover the rank-ordered ramp despite the per-rank size mismatch.
        local = torch.full((2, rank + 2), float(rank), device=device)
        parts = _all_gather_uneven(local, world)

        assert [tuple(p.shape) for p in parts] == [(2, r + 2) for r in range(size)]
        glued = torch.cat(parts, dim=1)
        expected = torch.cat(
            [torch.full((2, r + 2), float(r), device=device) for r in range(size)],
            dim=1,
        )
        assert torch.equal(glued, expected)
