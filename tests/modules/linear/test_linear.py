import warnings
from typing import Any

import pytest
import torch
import torch.distributed as dist

import jigsaw.convert as convert
import jigsaw.modules.linear as linear
import jigsaw.parallel as parallel
from jigsaw.gather import (
    gather_1d_results,
    gather_1d_results_list,
    gather_results,
)
from tests.parity import (
    ShardSpec,
    all_ranks_ok,
    assert_close,
    check_parity,
    gather_tensor,
    split_tensor,
)

pytestmark = [pytest.mark.dist_required]

# 2-D layout: (batch, seq, feature) with seq split over row and features over col.
# The weight is (out, in) with out split over row and in over col; the bias is split
# over col and replicated over row.
_DATA_SPEC = ShardSpec(row=1, col=2)
_PARAM_RULES = {"weight": ShardSpec(row=0, col=1), "bias": ShardSpec(col=0)}
# ``model_group == col_group``: the batch is split over row, where the weight and
# bias are replicated (attention's ``qkv`` / ``head_concat`` layout).
_COL_ONLY_DATA_SPEC = ShardSpec(row=0, col=2)
_COL_ONLY_PARAM_RULES = {"weight": ShardSpec(col=1), "bias": ShardSpec(col=0)}
_ROW_MESHES = [
    pytest.param("process_mesh_dict_2x2", marks=pytest.mark.dist(4), id="2x2"),
    pytest.param("process_mesh_dict_4x2", marks=pytest.mark.dist(8), id="4x2"),
]


def _distributed_from(
    serial: torch.nn.Linear,
    mesh: dict,
    device: torch.device,
    model_group: dist.ProcessGroup | None = None,
    param_rules: dict[str, ShardSpec] = _PARAM_RULES,
    comm: str | None = None,
) -> linear.DistributedLinear:
    """Build a ``DistributedLinear`` holding this rank's shards of ``serial``."""
    group = mesh["model"] if model_group is None else model_group
    layer = linear.DistributedLinear(
        serial.in_features,
        serial.out_features,
        device=device,
        model_group=group,
        col_group=mesh["col"],
        row_group=mesh["row"],
        bias=serial.bias is not None,
        config=None
        if comm is None
        else linear.LinearConfig(group, device=device, comm=comm),
    )
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(split_tensor(getattr(serial, name), param_rules[name], mesh))
    return layer


def _check_linear(
    mesh: dict,
    device: torch.device,
    in_dim: int,
    out_dim: int,
    input_shape: tuple[int, ...],
    bias: bool,
    model_group: dist.ProcessGroup | None = None,
    data_spec: ShardSpec = _DATA_SPEC,
    param_rules: dict[str, ShardSpec] = _PARAM_RULES,
    comm: str | None = None,
    **parity_kwargs: Any,
) -> None:
    """Forward and backward parity of a ``DistributedLinear`` against ``nn.Linear``."""
    torch.manual_seed(0)  # identical serial reference on every rank
    serial = torch.nn.Linear(in_dim, out_dim, bias=bias, device=device)

    check_parity(
        serial_module=serial,
        build_distributed=lambda module: _distributed_from(
            module, mesh, device, model_group, param_rules, comm
        ),
        input_shape=input_shape,
        input_spec=data_spec,
        output_spec=data_spec,
        param_rules={k: v for k, v in param_rules.items() if bias or k != "bias"},
        process_mesh_dict=mesh,
        device=device,
        **parity_kwargs,
    )


class TestParity:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize(("in_dim", "out_dim"), [(4, 4), (6, 2), (6, 10)])
    def test_2x2(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        bias: bool,
        in_dim: int,
        out_dim: int,
    ) -> None:
        """Seq split over row, features over col."""
        _check_linear(
            process_mesh_dict_2x2, device, in_dim, out_dim, (2, 4, in_dim), bias
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [False, True])
    def test_2x1x2(
        self, device: torch.device, process_mesh_dict_2x1x2: dict, bias: bool
    ) -> None:
        """Feature-only split on each of two dp replicas."""
        _check_linear(process_mesh_dict_2x1x2, device, 4, 4, (1, 10, 4), bias)

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("bias", [False, True])
    def test_4x2(
        self, device: torch.device, process_mesh_dict_4x2: dict, bias: bool
    ) -> None:
        """Four-way row split."""
        _check_linear(process_mesh_dict_4x2, device, 8, 4, (1, 4, 8), bias)

    @pytest.mark.dist(4)
    def test_domain_and_tensor_parallel(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Batch split over row (weights replicated there), tensor parallel over col."""
        mesh = process_mesh_dict_2x2
        _check_linear(
            mesh,
            device,
            10,
            10,
            (2, 10, 10),
            bias=True,
            model_group=mesh["col"],
            data_spec=ShardSpec(row=0, col=2),
            param_rules={"weight": ShardSpec(col=1), "bias": ShardSpec(col=0)},
        )


class TestSumBackward:
    """``out.sum().backward()`` hands the kernels an expanded, stride-0 gradient.

    The gradient-path ids name which of input and weight need a gradient, which
    picks the backward kernel: a frozen first layer or backbone takes the
    weight-only or input-only kernel instead of the fused one.
    """

    _MESHES = [
        pytest.param("process_mesh_dict_1x2", marks=pytest.mark.dist(2), id="1x2"),
        pytest.param("process_mesh_dict_2x1x2", marks=pytest.mark.dist(4), id="2x1x2"),
        pytest.param("process_mesh_dict_2x2", marks=pytest.mark.dist(4), id="2x2"),
        pytest.param("process_mesh_dict_4x2", marks=pytest.mark.dist(8), id="4x2"),
    ]
    _GRAD_PATHS = {
        "weight_only": {"input_requires_grad": False},
        "input_only": {"frozen": ("weight", "bias")},
        "both": {},
    }

    @pytest.mark.parametrize("comm", ["all_to_all", "isend", "batch_isend_irecv"])
    @pytest.mark.parametrize("path", _GRAD_PATHS)
    @pytest.mark.parametrize("mesh_name", _MESHES)
    def test_matches_torch(
        self,
        device: torch.device,
        request: pytest.FixtureRequest,
        mesh_name: str,
        path: str,
        comm: str,
    ) -> None:
        """Forward and backward parity with a summed loss."""
        if comm == "all_to_all" and device.type != "cuda":
            pytest.skip("all_to_all needs CUDA")
        mesh = request.getfixturevalue(mesh_name)
        _check_linear(
            mesh,
            device,
            8,
            8,
            (2, 4, 8),
            bias=True,
            comm=comm,
            sum_backward=True,
            **self._GRAD_PATHS[path],
        )

    @pytest.mark.dist(4)
    @pytest.mark.parametrize("comm", ["all_to_all", "isend", "batch_isend_irecv"])
    @pytest.mark.parametrize("path", _GRAD_PATHS)
    def test_weight_replicated_over_row(
        self, device: torch.device, process_mesh_dict_2x2: dict, path: str, comm: str
    ) -> None:
        """Attention's ``qkv`` / ``head_concat`` layout: a 2-way layer on 4 ranks."""
        if comm == "all_to_all" and device.type != "cuda":
            pytest.skip("all_to_all needs CUDA")
        mesh = process_mesh_dict_2x2
        _check_linear(
            mesh,
            device,
            8,
            8,
            (2, 4, 8),
            bias=True,
            model_group=mesh["col"],
            data_spec=ShardSpec(row=0, col=2),
            param_rules={"weight": ShardSpec(col=1), "bias": ShardSpec(col=0)},
            comm=comm,
            sum_backward=True,
            **self._GRAD_PATHS[path],
        )


class TestTraining:
    @pytest.mark.dist(4)
    def test_weights_match_after_many_steps(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Weights still match ``nn.Linear`` after many Adam steps."""
        mesh = process_mesh_dict_2x2
        torch.manual_seed(0)
        serial = torch.nn.Linear(6, 4, device=device)
        layer = _distributed_from(serial, mesh, device)
        opt = torch.optim.Adam(layer.parameters())
        ref_opt = torch.optim.Adam(serial.parameters())
        # Own generator: every rank draws the same global inputs.
        gen = torch.Generator(device=device).manual_seed(1)

        for _ in range(10):
            x = torch.randn(2, 4, 6, device=device, generator=gen)
            g = torch.randn(2, 4, 4, device=device, generator=gen)
            opt.zero_grad()
            ref_opt.zero_grad()
            serial(x).backward(g)
            layer(split_tensor(x, _DATA_SPEC, mesh)).backward(
                split_tensor(g, _DATA_SPEC, mesh)
            )
            opt.step()
            ref_opt.step()

        for name, p in layer.named_parameters():
            assert_close(
                f"{name} after training",
                gather_tensor(p.detach(), _PARAM_RULES[name], mesh, device, name),
                getattr(serial, name).detach(),
                1e-5,
                1e-5,
                device,
                mesh["model"],
            )

    @pytest.mark.parametrize("mesh_name", _ROW_MESHES)
    def test_row_replicas_stay_in_sync(
        self, device: torch.device, request: pytest.FixtureRequest, mesh_name: str
    ) -> None:
        """With ``model_group == col_group``, different data per row rank."""
        mesh = request.getfixturevalue(mesh_name)
        torch.manual_seed(0)
        serial = torch.nn.Linear(8, 8, device=device)
        layer = _distributed_from(
            serial, mesh, device, mesh["col"], _COL_ONLY_PARAM_RULES
        )
        opt = torch.optim.Adam(layer.parameters())
        ref_opt = torch.optim.Adam(serial.parameters())
        gen = torch.Generator(device=device).manual_seed(1)

        for _ in range(5):
            x = torch.randn(4, 3, 8, device=device, generator=gen)
            g = torch.randn(4, 3, 8, device=device, generator=gen)
            opt.zero_grad()
            ref_opt.zero_grad()
            serial(x).backward(g)
            layer(split_tensor(x, _COL_ONLY_DATA_SPEC, mesh)).backward(
                split_tensor(g, _COL_ONLY_DATA_SPEC, mesh)
            )
            opt.step()
            ref_opt.step()

        # ``gather_tensor`` also fails if the row replicas differ.
        for name, p in layer.named_parameters():
            assert_close(
                f"{name} after training",
                gather_tensor(
                    p.detach(), _COL_ONLY_PARAM_RULES[name], mesh, device, name
                ),
                getattr(serial, name).detach(),
                1e-5,
                1e-5,
                device,
                mesh["model"],
            )


class TestColumnOnlyLayout:
    """Convert, merge and reshard with ``model_group == col_group``."""

    @staticmethod
    def _replica_ranks(p: torch.Tensor) -> list[list[int]]:
        return [dist.get_process_group_ranks(g) for g in parallel.replica_groups(p)]

    def _check_merged(
        self,
        layer: linear.DistributedLinear,
        serial: torch.nn.Linear,
        mesh: dict,
        model_kw: str,
        device: torch.device,
    ) -> None:
        group = mesh["model"]
        weight, bias = layer.merge_weights(mesh, model_kw=model_kw)
        assert_close("merged weight", weight, serial.weight, 0.0, 0.0, device, group)
        assert_close("merged bias", bias, serial.bias, 0.0, 0.0, device, group)
        torch_layer = layer.convert_to_torch(mesh, model_kw=model_kw)
        assert_close(
            "converted weight",
            torch_layer.weight,
            serial.weight,
            0.0,
            0.0,
            device,
            group,
        )

    @pytest.mark.parametrize("mesh_name", _ROW_MESHES)
    def test_round_trip(
        self, device: torch.device, request: pytest.FixtureRequest, mesh_name: str
    ) -> None:
        """Convert, merge, reshard to the row x col layout and back."""
        mesh = request.getfixturevalue(mesh_name)
        col_mesh = {**mesh, "model": mesh["col"]}
        row_ranks = dist.get_process_group_ranks(mesh["row"])
        torch.manual_seed(0)
        serial = torch.nn.Linear(8, 8, device=device).requires_grad_(False)

        layer = linear.DistributedLinear.convert_to_jigsaw(
            serial, mesh, device, model_kw="col"
        )
        for name, p in layer.named_parameters():
            assert_close(
                f"converted {name}",
                p,
                split_tensor(getattr(serial, name), _COL_ONLY_PARAM_RULES[name], mesh),
                0.0,
                0.0,
                device,
                mesh["model"],
            )
        marked = self._replica_ranks(layer.weight) == [row_ranks]
        self._check_merged(layer, serial, mesh, "col", device)

        layer.reshard_parameters(col_mesh, mesh)
        unmarked = self._replica_ranks(layer.weight) == []
        self._check_merged(layer, serial, mesh, "model", device)
        x = torch.randn(4, 4, 8, device=device)
        assert_close(
            "output on row x col",
            layer(split_tensor(x, _DATA_SPEC, mesh)),
            split_tensor(serial(x), _DATA_SPEC, mesh),
            1e-5,
            1e-5,
            device,
            mesh["model"],
        )

        layer.reshard_parameters(mesh, col_mesh)
        remarked = self._replica_ranks(layer.weight) == [row_ranks]
        self._check_merged(layer, serial, mesh, "col", device)
        assert_close(
            "output on col only",
            layer(split_tensor(x, _COL_ONLY_DATA_SPEC, mesh)),
            split_tensor(serial(x), _COL_ONLY_DATA_SPEC, mesh),
            1e-5,
            1e-5,
            device,
            mesh["model"],
        )
        assert all_ranks_ok(marked and unmarked and remarked, device, mesh["model"]), (
            "weight replica mark over row does not follow the layout"
        )


class TestBiasInit:
    @pytest.mark.dist(4)
    def test_xwt_bias_init(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Ensure bias terms are synchronized across row groups."""
        local_rank = dist.get_rank(group=process_mesh_dict_2x2["model"])
        torch.manual_seed(local_rank)

        linlayer = linear.DistributedLinear(
            6,
            10,
            device=device,
            model_group=process_mesh_dict_2x2["model"],
            col_group=process_mesh_dict_2x2["col"],
            row_group=process_mesh_dict_2x2["row"],
            bias=True,
        ).to(device)

        assert isinstance(linlayer.bias, torch.Tensor)

        b_gathered = gather_1d_results_list(
            linlayer.bias.data, process_mesh_dict=process_mesh_dict_2x2
        )
        model_group = process_mesh_dict_2x2["model"]
        assert_close(
            "bias row replica 0/2",
            b_gathered[0],
            b_gathered[2],
            1e-5,
            1e-5,
            device,
            model_group,
        )
        assert_close(
            "bias row replica 1/3",
            b_gathered[1],
            b_gathered[3],
            1e-5,
            1e-5,
            device,
            model_group,
        )
        # ranks that are part of the same model group should have different biases.
        differ = not torch.allclose(b_gathered[0], b_gathered[1], atol=1e-5)
        assert all_ranks_ok(differ, device, model_group), (
            "bias must differ across the model group"
        )


class TestParameterCount:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("bias", [True, False])
    def test_count_is_global_on_every_rank(
        self, device: torch.device, process_mesh_dict_2x2: dict, bias: bool
    ) -> None:
        """Weight split over row and col, bias replicated over row: still global."""
        mesh = process_mesh_dict_2x2
        in_dim, out_dim = 6, 8
        layer = linear.DistributedLinear(
            in_dim,
            out_dim,
            device=device,
            model_group=mesh["model"],
            col_group=mesh["col"],
            row_group=mesh["row"],
            bias=bias,
        ).to(device)

        expected = in_dim * out_dim + (out_dim if bias else 0)
        assert all_ranks_ok(
            layer.get_parameter_count() == expected, device, mesh["model"]
        )


class TestDeprecatedLocalRank:
    @pytest.mark.dist(4)
    def test_local_rank_is_ignored(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """A stale ``local_rank`` warns and does not change the output."""
        mesh = process_mesh_dict_2x2
        torch.manual_seed(0)  # identical serial reference on every rank
        serial = torch.nn.Linear(4, 4, device=device)
        reference = _distributed_from(serial, mesh, device)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layer = linear.DistributedLinear(
                4,
                4,
                device=device,
                model_group=mesh["model"],
                col_group=mesh["col"],
                row_group=mesh["row"],
                local_rank=0,  # wrong on every rank but 0
            )
        warned = any(
            issubclass(w.category, DeprecationWarning)
            and "local_rank" in str(w.message)
            for w in caught
        )
        layer.load_state_dict(reference.state_dict())

        x = split_tensor(torch.randn(2, 4, 4, device=device), _DATA_SPEC, mesh)
        assert all_ranks_ok(warned, device, mesh["model"])
        assert_close("output", layer(x), reference(x), 0.0, 0.0, device, mesh["model"])


class TestComm:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize("comm", ["all_to_all", "isend", "batch_isend_irecv"])
    def test_every_comm_matches_torch(
        self, device: torch.device, process_mesh_dict_2x2: dict, comm: str
    ) -> None:
        """Forward and backward parity of each strategy; resharding keeps it."""
        if comm == "all_to_all" and device.type != "cuda":
            pytest.skip("all_to_all needs CUDA")
        mesh = process_mesh_dict_2x2
        model_group = mesh["model"]
        local_rank = dist.get_rank(group=model_group)
        torch.manual_seed(local_rank)

        in_dim, out_dim = 6, 4
        linlayer = linear.DistributedLinear(
            in_dim,
            out_dim,
            device=device,
            model_group=model_group,
            col_group=mesh["col"],
            row_group=mesh["row"],
            config=linear.LinearConfig(model_group, device=device, comm=comm),
        ).to(device)
        ref_linlayer = torch.nn.Linear(in_dim, out_dim).to(device)
        ref_linlayer.weight.data = gather_results(
            linlayer.weight.data, process_mesh_dict=mesh
        )
        assert isinstance(linlayer.bias, torch.Tensor)
        ref_linlayer.bias.data = gather_1d_results(
            linlayer.bias.data, mesh, dim=0, pm_key="col"
        )

        x_in = torch.randn(1, 2, in_dim // 2, device=device, requires_grad=True)
        x_gathered = gather_results(x_in.detach(), process_mesh_dict=mesh)
        x_gathered.requires_grad_(True)
        y = linlayer(x_in)
        ref_y = ref_linlayer(x_gathered)
        # Each rank holds a disjoint tile of the output: local sums add up to the global one.
        y.square().sum().backward()
        ref_y.square().sum().backward()

        assert isinstance(linlayer.weight.grad, torch.Tensor)
        assert isinstance(ref_linlayer.weight.grad, torch.Tensor)
        assert isinstance(x_in.grad, torch.Tensor)
        assert isinstance(x_gathered.grad, torch.Tensor)
        y_gathered = gather_results(y.detach(), process_mesh_dict=mesh)
        assert_close("output", y_gathered, ref_y, 1e-5, 1e-5, device, model_group)
        assert_close(
            "weight grad",
            gather_results(linlayer.weight.grad, process_mesh_dict=mesh),
            ref_linlayer.weight.grad,
            1e-5,
            1e-5,
            device,
            model_group,
        )
        assert_close(
            "input grad",
            gather_results(x_in.grad, process_mesh_dict=mesh),
            x_gathered.grad,
            1e-5,
            1e-5,
            device,
            model_group,
        )

        linlayer.reshard_parameters(mesh, mesh)
        assert all_ranks_ok(linlayer.config.comm == comm, device, model_group)


_CONVERT_MESHES = pytest.mark.parametrize(
    "mesh",
    [
        dict(row=1, col=2),
        dict(row=2, col=2),
        dict(row=4, col=2),
    ],
    indirect=True,
)


class TestConvert:
    """Options of ``convert_to_jigsaw`` reach the converted layer."""

    @_CONVERT_MESHES
    @pytest.mark.parametrize(
        ("dtype", "tol"),
        [
            pytest.param(torch.bfloat16, 2e-2, id="bf16"),
            pytest.param(torch.float64, 1e-12, id="fp64"),
        ],
    )
    def test_round_trip_keeps_dtype(
        self, device: torch.device, mesh: Any, dtype: torch.dtype, tol: float
    ) -> None:
        """Convert and convert back keep the source dtype and values."""
        group = mesh["model"]
        torch.manual_seed(0)
        serial = torch.nn.Linear(8, 8, device=device, dtype=dtype).requires_grad_(False)

        layer = linear.DistributedLinear.convert_to_jigsaw(serial, mesh, device)
        converted = layer.dtype == dtype and all(
            p.dtype == dtype for p in layer.parameters()
        )
        x = torch.randn(4, 4, 8, device=device, dtype=dtype)
        out = layer(split_tensor(x, _DATA_SPEC, mesh))
        assert_close(
            "output",
            out,
            split_tensor(serial(x), _DATA_SPEC, mesh),
            tol,
            tol,
            device,
            group,
        )

        back = layer.convert_to_torch(mesh)
        restored = out.dtype == dtype and all(
            p.dtype == dtype for p in back.parameters()
        )
        assert_close("weight", back.weight, serial.weight, 0.0, 0.0, device, group)
        assert_close("bias", back.bias, serial.bias, 0.0, 0.0, device, group)
        assert all_ranks_ok(converted and restored, device, group), (
            f"dtype not kept: layer {layer.dtype}, output {out.dtype}, "
            f"converted back {back.weight.dtype}, expected {dtype}"
        )

    @pytest.mark.parametrize("mesh", [dict(row=2, col=2)], indirect=True)
    def test_comm_reaches_layers(self, device: torch.device, mesh: Any) -> None:
        """``comm`` selects the transport, also through the model converter."""
        group = mesh["model"]
        # Not the default transport on either device.
        comm = "isend" if device.type == "cpu" else "batch_isend_irecv"
        torch.manual_seed(0)
        serial = torch.nn.Linear(8, 8, device=device).requires_grad_(False)

        layer = linear.DistributedLinear.convert_to_jigsaw(
            serial, mesh, device, comm=comm
        )
        x = torch.randn(4, 4, 8, device=device)
        assert_close(
            "output",
            layer(split_tensor(x, _DATA_SPEC, mesh)),
            split_tensor(serial(x), _DATA_SPEC, mesh),
            1e-5,
            1e-5,
            device,
            group,
        )

        model = torch.nn.ModuleDict(
            {"proj": torch.nn.Linear(8, 8, device=device), "mm": linear.MatMulXWT()}
        )
        model = convert.convert_to_jigsaw(model, mesh, device, comm=comm)
        comms = [layer.config.comm, model["proj"].config.comm, model["mm"].config.comm]
        assert all_ranks_ok(comms == [comm] * 3, device, group), (
            f"transports {comms}, expected {comm}"
        )

    @pytest.mark.parametrize("mesh", [dict(row=2, col=2)], indirect=True)
    def test_unknown_comm_raises(self, device: torch.device, mesh: Any) -> None:
        """An unknown ``comm`` raises on every rank before any communication."""
        serial = torch.nn.Linear(8, 8, device=device)
        with pytest.raises(ValueError, match="comm must be one of"):
            linear.DistributedLinear.convert_to_jigsaw(
                serial, mesh, device, comm="carrier_pigeon"
            )
        with pytest.raises(ValueError, match="comm must be one of"):
            linear.DistributedMatMulXWT.convert_to_jigsaw(
                linear.MatMulXWT(), mesh, device, comm="carrier_pigeon"
            )
