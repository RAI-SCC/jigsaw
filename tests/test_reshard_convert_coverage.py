"""Contract- and edge-case coverage for the sharding-conversion API.

``tests/test_sharding_configs.py`` sweeps the happy path -- every
``from_mesh -> to_mesh`` pair, every conv rank, ``bias`` on/off -- for the four
methods that move parameters between a plain ``torch.nn`` module and a jigsaw one
and between process meshes (``convert_to_jigsaw`` / ``convert_to_torch`` /
``merge_weights`` / ``reshard_parameters``). This file covers what that sweep
leaves out:

* **Divisibility contracts.** ``convert_to_jigsaw`` and ``reshard_parameters``
  each assert that the (global) parameter dims divide the target mesh; a mesh
  that does not divide must raise rather than silently mis-shard. Checked for
  linear, ``ConvNd``, ``ConvTransposeNd``, ``DistributedLayerNorm``,
  ``DistributedGroupNorm`` and ``Attention``.
* **Keyword overrides.** ``row_kw`` / ``col_kw`` / ``model_kw`` must thread
  through ``convert_to_jigsaw`` -> ``merge_weights`` -> ``reshard_parameters`` ->
  ``convert_to_torch`` for a mesh dict keyed with non-default names.
* **Generic-container recursion.** ``jigsaw.convert.convert_to_jigsaw`` swaps
  every mapped leaf inside an ``nn.Sequential`` / ``nn.Module`` and returns the
  same container object, leaving non-mapped leaves untouched; a module with no
  mapped leaf comes back unchanged.
* **Multi-hop reshard.** An ``A -> B -> C -> A`` chain preserves parameter
  values, not just a single hop.
* **dtype round-trip.** ``convert_to_jigsaw`` / ``convert_to_torch`` preserve a
  non-default weight dtype for conv.
"""

import copy

import pytest
import torch

import jigsaw.convert as convert
from jigsaw.modules import convolution, roll
from jigsaw.modules.attention import Attention
from jigsaw.modules.linear.linear import _SUPPORTED_XWT_LAYOUTS
from jigsaw.modules.normalization import normalization as norm
from jigsaw.utils_aggregate import gather_1d_results
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]


# --------------------------------------------------------------------------- #
# convert_to_jigsaw: divisibility contracts
# --------------------------------------------------------------------------- #
class TestConvertContracts:
    """A mesh axis that does not divide the parameter must raise on every rank.

    Each assertion lives before the first collective in the respective
    ``convert_to_jigsaw``, so every rank raises identically and the job fails
    instead of hanging.
    """

    @pytest.mark.dist(4)
    def test_linear_in_features_indivisible_by_row(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        lin = torch.nn.Linear(3, 4, device=device)  # in=3 not divisible by row=2
        with pytest.raises(AssertionError):
            convert.convert_to_jigsaw(lin, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_linear_out_features_indivisible_by_col(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        lin = torch.nn.Linear(4, 3, device=device)  # out=3 not divisible by col=2
        with pytest.raises(AssertionError):
            convert.convert_to_jigsaw(lin, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_conv_in_channels_indivisible_by_col(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        conv = torch.nn.Conv2d(3, 4, 2, device=device)  # in=3 not divisible by col=2
        with pytest.raises(AssertionError):
            convert.convert_to_jigsaw(conv, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_conv_transpose_out_channels_indivisible_by_col(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        # ConvTransposeNd shards out_channels over col; out=3 not divisible by 2.
        convt = torch.nn.ConvTranspose2d(4, 3, 2, device=device)
        with pytest.raises(AssertionError):
            convert.convert_to_jigsaw(convt, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_attention_heads_indivisible_by_channel(
        self, device: torch.device, process_mesh_dict_2x1x2: dict
    ) -> None:
        # 2x1x2 -> row=1, col=2; heads=3 is not divisible by the channel size.
        attn = Attention(in_dim=12, embed_dim=12, out_dim=12, heads=3, device=device)
        with pytest.raises(AssertionError):
            Attention.convert_to_jigsaw(attn, process_mesh_dict_2x1x2, device)


# --------------------------------------------------------------------------- #
# reshard_parameters: divisibility contracts
# --------------------------------------------------------------------------- #
class TestReshardContracts:
    """``reshard_parameters`` must reject a target mesh that cannot hold the
    gathered global parameter, and must do so before it gathers anything."""

    @pytest.mark.dist(4)
    def test_linear_reshard_row_indivisible(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_4x1: dict,
    ) -> None:
        jm = convert.convert_to_jigsaw(
            torch.nn.Linear(2, 2, device=device), process_mesh_dict_2x2, device
        )
        # global weight is (2, 2); target row=4 cannot slice a size-2 axis.
        with pytest.raises(AssertionError):
            jm.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_4x1)

    @pytest.mark.dist(4)
    def test_conv_reshard_col_indivisible(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        jm = convert.convert_to_jigsaw(
            torch.nn.Conv2d(2, 2, 2, device=device), process_mesh_dict_2x2, device
        )
        # global out_channels is 2; target col=4 cannot slice it.
        with pytest.raises(AssertionError):
            jm.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)

    @pytest.mark.dist(4)
    def test_conv_transpose_reshard_col_indivisible(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        jm = convert.convert_to_jigsaw(
            torch.nn.ConvTranspose2d(2, 2, 2, device=device),
            process_mesh_dict_2x2,
            device,
        )
        with pytest.raises(AssertionError):
            jm.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)

    @pytest.mark.dist(4)
    def test_layer_norm_reshard_dim_indivisible(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        # local_dim=3 over col=2 -> global normalized dim 6, not divisible by col=4.
        ln = norm.DistributedLayerNorm(
            local_dim=3,
            local_rank=process_mesh_dict_2x2["model"].rank(),
            device=device,
            row_group=process_mesh_dict_2x2["row"],
            col_group=process_mesh_dict_2x2["col"],
        )
        with pytest.raises(AssertionError):
            ln.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)

    @pytest.mark.dist(4)
    def test_group_norm_reshard_groups_indivisible(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        # 1 local group over col=2 -> 2 global groups, not divisible by col=4.
        gn = norm.DistributedGroupNorm(
            num_local_groups=1,
            num_channels=3,
            device=device,
            channel_group=process_mesh_dict_2x2["col"],
        )
        with pytest.raises(AssertionError):
            gn.reshard_parameters(process_mesh_dict_2x2, process_mesh_dict_1x4)


# --------------------------------------------------------------------------- #
# row_kw / col_kw / model_kw overrides
# --------------------------------------------------------------------------- #
class TestKeywordOverrides:
    """The whole convert / merge / reshard / convert-back cycle must work on a
    mesh dict whose groups are keyed with names other than row/col/model."""

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [True, False])
    def test_alias_keyed_mesh_round_trip(
        self, device: torch.device, process_mesh_dict_2x2: dict, bias: bool
    ) -> None:
        m = process_mesh_dict_2x2
        alias = {"tok": m["row"], "chan": m["col"], "mp": m["model"]}
        kw = dict(row_kw="tok", col_kw="chan", model_kw="mp")
        grp = alias["mp"]

        torch.manual_seed(1234)  # every rank must build the same reference
        ref = torch.nn.Linear(24, 16, bias=bias, device=device)
        jm = convert.convert_to_jigsaw(copy.deepcopy(ref), alias, device, **kw)

        w, b = jm.merge_weights(alias, "tok", "chan", "mp")
        assert_close("merged weight", w, ref.weight.data, 1e-6, 1e-6, device, grp)
        if bias:
            assert_close("merged bias", b, ref.bias.data, 1e-6, 1e-6, device, grp)
        else:
            assert all_ranks_ok(b is None, device, grp)

        # reshard onto the same (alias-keyed) mesh, then convert straight back.
        jm.reshard_parameters(alias, alias, "tok", "chan", "mp")
        back = jm.convert_to_torch(alias, "tok", "chan", "mp")
        assert_close(
            "weight", back.weight.data, ref.weight.data, 1e-6, 1e-6, device, grp
        )
        if bias:
            assert_close("bias", back.bias.data, ref.bias.data, 1e-6, 1e-6, device, grp)
        else:
            assert all_ranks_ok(back.bias is None, device, grp)


# --------------------------------------------------------------------------- #
# jigsaw.convert.convert_to_jigsaw: generic-container recursion
# --------------------------------------------------------------------------- #
class TestGenericContainerRecursion:
    @pytest.mark.dist(4)
    def test_sequential_leaves_swapped_in_place(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        from jigsaw.modules import convolution, linear

        seq = torch.nn.Sequential(
            torch.nn.Linear(8, 8, device=device),
            torch.nn.ReLU(),
            torch.nn.Conv1d(8, 8, 2, stride=2, device=device),
        )
        relu = seq[1]
        out = convert.convert_to_jigsaw(seq, process_mesh_dict_2x2, device)

        assert out is seq, "generic container is converted in place and returned"
        assert isinstance(seq[0], linear.DistributedXWT)
        assert seq[1] is relu, "a non-mapped leaf is left untouched"
        assert isinstance(seq[2], convolution.DistributedConv1D)

    @pytest.mark.dist(4)
    def test_module_without_mapped_leaf_unchanged(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        act = torch.nn.GELU()
        assert convert.convert_to_jigsaw(act, process_mesh_dict_2x2, device) is act, (
            "a module with no mapped leaf is returned as-is"
        )


# --------------------------------------------------------------------------- #
# multi-hop reshard: A -> B -> C -> A preserves parameter values
# --------------------------------------------------------------------------- #
class TestMultiHopReshard:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [True, False])
    def test_linear_chain(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        bias: bool,
    ) -> None:
        a = request.getfixturevalue("process_mesh_dict_2x2")
        b = request.getfixturevalue("process_mesh_dict_4x1x1")
        c = request.getfixturevalue("process_mesh_dict_1x4")

        torch.manual_seed(1234)
        ref = torch.nn.Linear(24, 16, bias=bias, device=device)
        jm = convert.convert_to_jigsaw(copy.deepcopy(ref), a, device)

        jm.reshard_parameters(a, b)
        jm.reshard_parameters(b, c)
        jm.reshard_parameters(c, a)

        grp = a["model"]
        w, bvec = jm.merge_weights(a)
        assert_close(
            "weight after A->B->C->A", w, ref.weight.data, 1e-6, 1e-6, device, grp
        )
        if bias:
            assert_close(
                "bias after A->B->C->A", bvec, ref.bias.data, 1e-6, 1e-6, device, grp
            )

        # landed back on a layout the XWT kernel supports -> a forward still runs.
        if (a["row"].size(), a["col"].size()) in _SUPPORTED_XWT_LAYOUTS:
            x = torch.randn(2, 5, jm.weight.shape[1], device=device)
            y = jm(x)
            assert all_ranks_ok(y is not None, device, grp)

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv_chain(
        self,
        request: pytest.FixtureRequest,
        device: torch.device,
        bias: bool,
    ) -> None:
        a = request.getfixturevalue("process_mesh_dict_2x2")
        b = request.getfixturevalue("process_mesh_dict_4x1")
        c = request.getfixturevalue("process_mesh_dict_2x2x1")

        torch.manual_seed(1234)
        ref = torch.nn.Conv2d(24, 16, 3, stride=3, bias=bias, device=device)
        jm = convert.convert_to_jigsaw(copy.deepcopy(ref), a, device)

        jm.reshard_parameters(a, b)
        jm.reshard_parameters(b, c)
        jm.reshard_parameters(c, a)

        grp = a["model"]
        w, bvec = jm.merge_weights(a, "col")
        assert_close(
            "weight after A->B->C->A", w, ref.weight.data, 1e-6, 1e-6, device, grp
        )
        if bias:
            assert_close(
                "bias after A->B->C->A", bvec, ref.bias.data, 1e-6, 1e-6, device, grp
            )


# --------------------------------------------------------------------------- #
# dtype survives the torch <-> jigsaw round-trip (conv)
# --------------------------------------------------------------------------- #
class TestDtypeRoundTrip:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [True, False])
    def test_conv_float64_round_trip(
        self, device: torch.device, process_mesh_dict_2x2: dict, bias: bool
    ) -> None:
        mesh = process_mesh_dict_2x2
        grp = mesh["model"]
        torch.manual_seed(1234)  # every rank must build the same reference
        ref = torch.nn.Conv2d(
            8, 8, 2, stride=2, bias=bias, device=device, dtype=torch.float64
        )
        jm = convert.convert_to_jigsaw(copy.deepcopy(ref), mesh, device)
        assert all_ranks_ok(jm.weight.dtype == torch.float64, device, grp)

        back = jm.convert_to_torch(mesh)
        assert all_ranks_ok(back.weight.dtype == torch.float64, device, grp)
        assert_close(
            "weight", back.weight.data, ref.weight.data, 1e-12, 1e-12, device, grp
        )
        if bias:
            assert all_ranks_ok(back.bias.dtype == torch.float64, device, grp)
            assert_close(
                "bias", back.bias.data, ref.bias.data, 1e-12, 1e-12, device, grp
            )


# --------------------------------------------------------------------------- #
# convert_to_jigsaw: mesh axes, devices, forwarded options
# --------------------------------------------------------------------------- #
class TestConvertAxesDevicesOptions:
    @pytest.mark.dist(4)
    def test_linear_divisibility_follows_the_shard_axes(
        self, device: torch.device, process_mesh_dict_4x1: dict
    ) -> None:
        """The columns shard ``in_features``, the rows ``out_features``."""
        mesh = process_mesh_dict_4x1
        torch.manual_seed(0)  # every rank must build the same reference
        ref = torch.nn.Linear(3, 4, device=device)  # in=3 need not divide row=4
        jm = convert.convert_to_jigsaw(copy.deepcopy(ref), mesh, device)
        w, b = jm.merge_weights(mesh)
        group = mesh["model"]
        assert_close("merged weight", w, ref.weight.data, 1e-6, 1e-6, device, group)
        assert_close("merged bias", b, ref.bias.data, 1e-6, 1e-6, device, group)

        lin = torch.nn.Linear(4, 3, device=device)  # out=3 not divisible by row=4
        with pytest.raises(AssertionError, match="rows"):
            convert.convert_to_jigsaw(lin, mesh, device)

    @pytest.mark.dist(4)
    def test_linear_from_cpu_lands_on_the_device(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Weight and bias of a CPU layer both move to the target device."""
        torch.manual_seed(0)
        jm = convert.convert_to_jigsaw(
            torch.nn.Linear(8, 8), process_mesh_dict_2x2, device
        )
        assert jm.weight.device.type == device.type
        assert jm.bias.device == jm.weight.device
        assert jm(torch.randn(2, 3, 4, device=device)).shape == (2, 3, 4)

    @pytest.mark.dist(4)
    def test_options_of_one_converter_reach_a_mixed_model(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """``ring`` is an attention option; the other layers must accept it."""
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Conv2d(4, 4, 2, 2, device=device),
            torch.nn.ConvTranspose2d(4, 4, 2, 2, device=device),
            roll.Roll(shifts=(1, 1), dims=(-3, -2)),
            torch.nn.Linear(4, 4, device=device),
            Attention(8, 8, 8, 2, device),
        )
        out = convert.convert_to_jigsaw(
            model, process_mesh_dict_2x2, device, ring=False
        )
        assert [type(m).__name__ for m in out] == [
            "DistributedConv2D",
            "DistributedConv2DTranspose",
            "DistributedRoll",
            "DistributedXWT",
            "DistributedAttention",
        ]
        assert out[4].ring is False

    @pytest.mark.dist(4)
    def test_roll_with_dims_counted_from_the_front(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """The default split of dimension ``-2`` matches ``dims=(1, 2)`` in 4-D."""
        mesh = process_mesh_dict_2x2
        jr = convert.convert_to_jigsaw(
            roll.Roll(shifts=(2, 3), dims=(1, 2)), mesh, device
        )
        torch.manual_seed(mesh["model"].rank())
        x = torch.randn(1, 8, 8, 16, device=device)
        x_global = gather_1d_results(x, mesh, dim=2, pm_key="row")
        y = gather_1d_results(jr(x), mesh, dim=2, pm_key="row")
        ref = torch.roll(x_global, shifts=(2, 3), dims=(1, 2))
        assert_close("roll forward", y, ref, 1e-6, 1e-6, device, mesh["model"])

    @pytest.mark.dist(4)
    def test_roll_default_split_uses_row_kw(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        m = process_mesh_dict_2x2
        alias = {"tok": m["row"], "chan": m["col"], "mp": m["model"]}
        jr = convert.convert_to_jigsaw(
            roll.Roll(shifts=(1,), dims=(-2,)),
            alias,
            device,
            row_kw="tok",
            col_kw="chan",
            model_kw="mp",
        )
        assert jr.spatial_groups == (m["row"],)


# --------------------------------------------------------------------------- #
# dtype of freshly built layers
# --------------------------------------------------------------------------- #
class TestConstructorDtype:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "cls", [convolution.DistributedConv2D, convolution.DistributedConv2DTranspose]
    )
    def test_conv_parameters_have_the_requested_dtype(
        self, device: torch.device, process_mesh_dict_2x2: dict, cls: type
    ) -> None:
        mesh = process_mesh_dict_2x2
        layer = cls(4, 4, 3, device, mesh["row"], mesh["col"], dtype=torch.float64)
        assert layer.weight.dtype == torch.float64
        assert layer.bias.dtype == torch.float64

    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_serial_attention_has_the_requested_dtype(
        self, device: torch.device, qk_norm: bool
    ) -> None:
        attn = Attention(8, 8, 8, 2, device, qk_norm=qk_norm, dtype=torch.float64)
        assert {p.dtype for p in attn.parameters()} == {torch.float64}
        assert {p.device.type for p in attn.parameters()} == {device.type}
        x = torch.randn(2, 6, 8, device=device, dtype=torch.float64)
        assert attn(x).dtype == torch.float64
