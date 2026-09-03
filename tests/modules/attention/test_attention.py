import pytest
import torch

import jigsaw.modules.attention as attention
from jigsaw.utils_aggregate import (
    gather_1d_results,
    gather_1d_results_list,
)
from tests.parity import all_ranks_ok, assert_close

pytestmark = [pytest.mark.dist_required]

_IN_DIM = _EMBED_DIM = _OUT_DIM = 256
_HEADS = 4


def _build_distributed(
    mesh: dict, device: torch.device, **overrides: object
) -> attention.DistributedAttention:
    """Construct a ``DistributedAttention`` on ``mesh`` with the shared test shape."""
    kwargs: dict = dict(
        in_dim=_IN_DIM,
        embed_dim=_EMBED_DIM,
        out_dim=_OUT_DIM,
        heads=_HEADS,
        dropout_p=0.0,
        local_rank=mesh["col"].rank(),
        device=device,
        spatial_group=mesh["row"],
        channel_group=mesh["col"],
        bayesian=False,
        qk_norm=False,
        ring_attention=False,
    )
    kwargs.update(overrides)
    return attention.DistributedAttention(**kwargs)


def _assert_shard_replication(
    shards: list[torch.Tensor], name: str, device: torch.device, model_group: object
) -> None:
    """A model-parallel parameter agrees across the spatial group and differs across the channel group.

    ``shards`` holds one tensor per model-group rank; a row-major 2x2 mesh puts
    rank ``r`` at ``(row, col) = divmod(r, 2)``, so shards 0 and 2 share a channel
    column -- they must match after the spatial broadcast + grad-sync -- while
    shards 0 and 1 sit in different columns and must differ. Routed through the
    parity helpers so a mismatch on a subset of ranks fails the job instead of
    hanging the rest in the next collective.
    """
    assert_close(
        f"{name} across the spatial group",
        shards[0],
        shards[2],
        1e-8,
        1e-5,
        device,
        model_group,
    )
    differ = not torch.allclose(shards[0], shards[1])
    assert all_ranks_ok(differ, device, model_group), (
        f"{name} must differ across the channel group"
    )


def _assert_all_params_replicated(
    model: attention.DistributedAttention, mesh: dict, device: torch.device
) -> None:
    """Run :func:`_assert_shard_replication` on every model-parallel parameter."""
    model_group = mesh["model"]
    named: dict[str, torch.Tensor | None] = {
        "qkv weight": model.qkv.weight,
        "head_concat weight": model.head_concat.weight,
        "head_concat bias": model.head_concat.bias,
    }
    for name, param in named.items():
        assert param is not None, f"{name} is missing"
        shards = gather_1d_results_list(param, process_mesh_dict=mesh, pm_key="model")
        _assert_shard_replication(shards, name, device, model_group)


class TestDistributedAttention:
    @pytest.mark.dist(4)
    def test_init_attention(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Freshly built parameters are already synchronized across the spatial group."""
        model = _build_distributed(process_mesh_dict_2x2, device)
        assert isinstance(model, attention.DistributedAttention)
        _assert_all_params_replicated(model, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_attention_sync(
        self, device: torch.device, process_mesh_dict_2x2: dict
    ) -> None:
        """Ten optimizer steps keep the spatial replicas locked (the grad-sync hook fired)."""
        model = _build_distributed(process_mesh_dict_2x2, device)
        batch, local_seq_len = 10, 14
        local_channels = _IN_DIM // process_mesh_dict_2x2["col"].size()
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(10):
            x = torch.randn(batch, local_seq_len, local_channels, device=device)
            pred = model(x)
            y = torch.randn_like(pred)
            loss = loss_fn(y, pred)
            loss.backward()
            optimizer.step()

        _assert_all_params_replicated(model, process_mesh_dict_2x2, device)

    @pytest.mark.dist(4)
    def test_converter_parity(
        self, device: torch.device, process_mesh_dict_2x1x2: dict
    ) -> None:
        """Serial ``Attention`` -> ``convert_to_jigsaw`` reproduces the forward, the input
        gradient and the ``head_concat`` weight/gradient.

        Pure channel split (spatial size 1), ring off, so the sharded layer is
        equivalent to a plain serial call. ``qkv`` is compared by sum because the
        converter permutes its ``[3, heads, head_dim, in]`` layout, so a plain
        gather does not line the elements back up.
        """
        mesh = process_mesh_dict_2x1x2
        col = mesh["col"]
        model_group = mesh["model"]

        torch.manual_seed(1)  # identical serial reference on every rank
        serial = attention.Attention(
            _IN_DIM, _EMBED_DIM, _OUT_DIM, _HEADS, device, dropout_p=0.0
        )
        ref_qkv_w = serial.qkv.weight.detach().clone()
        ref_hc_w = serial.head_concat.weight.detach().clone()

        x_global = torch.randn(1, 10, _IN_DIM, device=device, requires_grad=True)
        y_ref = serial(x_global)
        y_ref.backward(torch.ones_like(y_ref))
        assert x_global.grad is not None
        assert serial.qkv.weight.grad is not None
        assert serial.head_concat.weight.grad is not None
        ref_out = y_ref.detach().clone()
        ref_x_grad = x_global.grad.detach().clone()
        ref_qkv_grad = serial.qkv.weight.grad.detach().clone()
        ref_hc_grad = serial.head_concat.weight.grad.detach().clone()

        dist_model = attention.Attention.convert_to_jigsaw(
            serial, mesh, device, ring=False
        )

        cshard = _IN_DIM // col.size()
        x_local = (
            x_global.detach()[..., col.rank() * cshard : (col.rank() + 1) * cshard]
            .clone()
            .requires_grad_(True)
        )
        y_local = dist_model(x_local)
        y_local.backward(torch.ones_like(y_local))

        assert x_local.grad is not None
        assert dist_model.qkv.weight.grad is not None
        assert dist_model.head_concat.weight.grad is not None

        out = gather_1d_results(y_local, mesh, dim=-1, pm_key="col")
        x_grad = gather_1d_results(x_local.grad, mesh, dim=-1, pm_key="col")
        hc_w = gather_1d_results(
            dist_model.head_concat.weight.detach(), mesh, dim=-1, pm_key="col"
        )
        hc_grad = gather_1d_results(
            dist_model.head_concat.weight.grad, mesh, dim=-1, pm_key="col"
        )
        qkv_w = gather_1d_results(
            dist_model.qkv.weight.detach(), mesh, dim=-1, pm_key="col"
        )
        qkv_grad = gather_1d_results(
            dist_model.qkv.weight.grad, mesh, dim=-1, pm_key="col"
        )

        assert_close("output", out, ref_out, 1e-4, 1e-4, device, model_group)
        assert_close("input grad", x_grad, ref_x_grad, 1e-4, 1e-4, device, model_group)
        assert_close(
            "head_concat weight", hc_w, ref_hc_w, 1e-5, 1e-5, device, model_group
        )
        assert_close(
            "head_concat grad", hc_grad, ref_hc_grad, 1e-4, 1e-4, device, model_group
        )
        assert_close(
            "qkv weight sum",
            qkv_w.sum(),
            ref_qkv_w.sum(),
            1e-4,
            1e-4,
            device,
            model_group,
        )
        assert_close(
            "qkv grad sum",
            qkv_grad.sum(),
            ref_qkv_grad.sum(),
            1e-4,
            1e-4,
            device,
            model_group,
        )

    @pytest.mark.dist(4)
    def test_rejects_bad_config(
        self,
        device: torch.device,
        process_mesh_dict_2x2: dict,
        process_mesh_dict_1x4: dict,
    ) -> None:
        """Construction fails fast on an indivisible head count or ring-without-spatial."""
        with pytest.raises(AssertionError, match="heads is not divisible"):
            _build_distributed(process_mesh_dict_2x2, device, heads=3)
        with pytest.raises(AssertionError, match="Cannot perform ring attention"):
            _build_distributed(process_mesh_dict_1x4, device, ring_attention=True)
