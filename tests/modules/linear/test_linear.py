import pytest
import torch
import torch.distributed as dist

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


def _distributed_from(
    serial: torch.nn.Linear,
    mesh: dict,
    device: torch.device,
    model_group: dist.ProcessGroup | None = None,
    param_rules: dict[str, ShardSpec] = _PARAM_RULES,
) -> linear.DistributedLinear:
    """Build a ``DistributedLinear`` holding this rank's shards of ``serial``."""
    layer = linear.DistributedLinear(
        serial.in_features,
        serial.out_features,
        local_rank=dist.get_rank(group=mesh["model"]),
        device=device,
        model_group=mesh["model"] if model_group is None else model_group,
        col_group=mesh["col"],
        row_group=mesh["row"],
        bias=serial.bias is not None,
    )
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(split_tensor(getattr(serial, name), param_rules[name], mesh))
    if model_group is not None:
        # row is outside the model group, so the weight is a row replica.
        parallel.mark_replicated(layer.weight, mesh["row"])
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
) -> None:
    """Forward and backward parity of a ``DistributedLinear`` against ``nn.Linear``."""
    torch.manual_seed(0)  # identical serial reference on every rank
    serial = torch.nn.Linear(in_dim, out_dim, bias=bias, device=device)

    check_parity(
        serial_module=serial,
        build_distributed=lambda module: _distributed_from(
            module, mesh, device, model_group, param_rules
        ),
        input_shape=input_shape,
        input_spec=data_spec,
        output_spec=data_spec,
        param_rules={k: v for k, v in param_rules.items() if bias or k != "bias"},
        process_mesh_dict=mesh,
        device=device,
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
            local_rank=local_rank,
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
            local_rank=dist.get_rank(group=mesh["model"]),
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
            local_rank=local_rank,
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
