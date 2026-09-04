import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention.config as config
from jigsaw.modules.attention.functional import backward, forward, functional
from jigsaw.utils_aggregate import (
    gather_1d_results,
)
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]


class TestHybridTopology:
    @pytest.mark.dist(8)
    @pytest.mark.parametrize("gpus_per_node", [1, 2, 4, 8])
    def test_get_hybrid_topology(
        self, device: torch.device, gpus_per_node: int
    ) -> None:
        """Split the world ring into intra-/inter-node sub-groups.

        Parameters
        ----------
        device : torch.device
            Device used to initialize the process group.
        gpus_per_node : int
            Number of consecutive ring ranks that share a node.
        """
        group = dist.group.WORLD
        assert group is not None
        world_size = dist.get_world_size(group)
        rank = dist.get_rank(group)

        functional._HYBRID_TOPOLOGY_CACHE.pop((group, gpus_per_node), None)
        topo = functional.get_hybrid_topology(group, gpus_per_node)

        assert topo.gpus_per_node == gpus_per_node
        assert topo.n_nodes == world_size // gpus_per_node
        assert (topo.node_idx, topo.local_idx) == divmod(rank, gpus_per_node)
        assert topo.ring_rank == rank

        assert dist.get_world_size(topo.intranode_group) == gpus_per_node
        assert dist.get_world_size(topo.internode_group) == topo.n_nodes

        # Intra-node peers are the contiguous chunk; inter-node peers share the
        # local offset across nodes.
        node_start = topo.node_idx * gpus_per_node
        assert sorted(dist.get_process_group_ranks(topo.intranode_group)) == list(
            range(node_start, node_start + gpus_per_node)
        )
        assert sorted(dist.get_process_group_ranks(topo.internode_group)) == list(
            range(topo.local_idx, world_size, gpus_per_node)
        )

        # Cached: same object on the second call.
        assert functional.get_hybrid_topology(group, gpus_per_node) is topo

    @pytest.mark.dist(8)
    @pytest.mark.parametrize("gpus_per_node", [1, 2, 4])
    def test_get_hybrid_topology_on_mesh_subgroup(
        self, device: torch.device, gpus_per_node: int
    ) -> None:
        """Run the ring over one mesh dimension, not the global world.

        With a ``(row=4, col=2)`` mesh the ring is the ``row`` group: its world
        size is 4 (not 8), and the two ``col`` rings build their sub-groups
        independently over their own global ranks.

        Parameters
        ----------
        device : torch.device
            Device used to initialize the process mesh.
        gpus_per_node : int
            Number of consecutive ring ranks that share a node.
        """
        mesh = dist.device_mesh.init_device_mesh(
            device_type=device.type,
            mesh_shape=(4, 2),
            mesh_dim_names=("row", "col"),
        )
        group = mesh.get_group("row")
        ring_global_ranks = dist.get_process_group_ranks(group)
        world_size = len(ring_global_ranks)
        assert world_size == 4  # ring size == size(row), not the global world size

        functional._HYBRID_TOPOLOGY_CACHE.pop((group, gpus_per_node), None)
        topo = functional.get_hybrid_topology(group, gpus_per_node)

        ring_pos = dist.get_rank(group)
        node_idx, local_idx = divmod(ring_pos, gpus_per_node)
        n_nodes = world_size // gpus_per_node

        assert (topo.n_nodes, topo.node_idx, topo.local_idx) == (
            n_nodes,
            node_idx,
            local_idx,
        )
        assert dist.get_world_size(topo.intranode_group) == gpus_per_node
        assert dist.get_world_size(topo.internode_group) == n_nodes

        node_start = node_idx * gpus_per_node
        assert sorted(dist.get_process_group_ranks(topo.intranode_group)) == sorted(
            ring_global_ranks[node_start : node_start + gpus_per_node]
        )
        assert sorted(dist.get_process_group_ranks(topo.internode_group)) == sorted(
            ring_global_ranks[node * gpus_per_node + local_idx]
            for node in range(n_nodes)
        )
        # Every member of a sub-group lands on the same underlying group.
        assert functional.get_hybrid_topology(group, gpus_per_node) is topo

    @pytest.mark.dist(4)
    def test_get_hybrid_topology_rejects_bad_args(self, device: torch.device) -> None:
        """Reject a non-positive or non-dividing ``gpus_per_node``.

        Parameters
        ----------
        device : torch.device
            Device used to initialize the process group.
        """
        group = dist.group.WORLD
        assert group is not None

        with pytest.raises(ValueError, match="gpus_per_node must be positive"):
            functional.get_hybrid_topology(group, 0)
        with pytest.raises(ValueError, match="divisible by gpus_per_node"):
            functional.get_hybrid_topology(group, 3)


class TestRingAttention:
    @pytest.mark.dist(4)
    @pytest.mark.parametrize(
        "send_in_forward", [config.SendForward.X, config.SendForward.KV]
    )
    @pytest.mark.parametrize("save_for_backward", [config.SaveBackward.KV])
    @pytest.mark.parametrize(
        "send_op",
        [config.SendOp.ALLTOALL, config.SendOp.P2P, config.SendOp.BROADCAST],
    )
    @pytest.mark.parametrize(
        "save_scope",
        [config.SaveScope.LOCAL, config.SaveScope.GLOBAL],
    )
    @pytest.mark.parametrize(
        "send_backward",
        [config.SendBackward.GATHER_DKV, config.SendBackward.SEND_DKV],
    )
    @pytest.mark.parametrize("offloader", [config.Offload.DISABLED, config.Offload.CPU])
    def test_ring_forward_backward(
        self,
        device: torch.device,
        process_mesh_dict_4x1: dict,
        send_in_forward: config.SendForward,
        save_for_backward: config.SaveBackward,
        send_op: config.SendOp,
        save_scope: config.SaveScope,
        send_backward: config.SendBackward,
        offloader: config.Offload,
    ) -> None:
        """Verify if forward-backward pass works for ring attention."""
        if (
            send_in_forward == config.SendForward.KV
            and save_for_backward == config.SaveBackward.X
        ):
            pytest.xfail("not yet supported")

        process_mesh_dict = process_mesh_dict_4x1
        ring_config = config.RingAttentionConfig(
            device=device,
            send_in_forward=send_in_forward,
            send_op=send_op,
            save_for_backward=save_for_backward,
            save_scope=save_scope,
            send_backward=send_backward,
            offload=offloader,
        )
        print(f"R{dist.get_rank()} config {ring_config}")

        ring_forward = forward.build_ring_forward(ring_config)
        local_rank = dist.get_rank(group=process_mesh_dict["row"])
        world_size = dist.get_world_size(group=process_mesh_dict["row"])
        dtype = torch.bfloat16 if dist.get_backend() == "nccl" else torch.float32
        torch.manual_seed(2)  # same seed for weight initialization
        b, h, n_global, d = 3, 6, world_size * 16, 16
        n_local = n_global // world_size

        w_qkv = torch.randn(
            3 * d * h, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        torch.manual_seed(local_rank)  # different seed for input data generation
        x = torch.randn(
            b, n_local, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        x.fill_(local_rank * 0.1)
        x = x.contiguous()

        qkv = x @ w_qkv.mT
        # QKV shape: [B, Nlocal, 3*d*h]
        # We need QKV to have shape: []
        q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(
            2, 0, 3, 1, 4
        )

        q_local, k_local, v_local = (
            q_local.contiguous(),
            k_local.contiguous(),
            v_local.contiguous(),
        )
        scale = q_local.shape[-1] ** -0.5

        # just test helper function forward
        ring_forward = forward.build_ring_forward(ring_config)
        out, lse, chunks_to_save = ring_forward(
            q_local,
            k_local,
            v_local,
            x,
            w_qkv,
            scale,
            0.0,
            process_mesh_dict["row"],
            local_rank,
            world_size,
            ring_config,
        )
        out_full = gather_1d_results(out, process_mesh_dict, dim=2)
        x_full = gather_1d_results(x, process_mesh_dict, dim=1)
        qkv = x_full @ w_qkv.mT
        q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )

        # Reference calculation
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=False, dropout_p=0.0
            )

        q_ref2, k_ref2, v_ref2 = (
            q_ref.detach().clone(),
            k_ref.detach().clone(),
            v_ref.detach().clone(),
        )
        q_ref2 = q_ref2.requires_grad_(True)
        k_ref2 = k_ref2.requires_grad_(True)
        v_ref2 = v_ref2.requires_grad_(True)

        out_core = ring_config.attn_core_forward_fn(
            q_ref2, k_ref2, v_ref2, dropout_p=0.0, scale=scale
        )[0]
        out_core = out_core.requires_grad_(True)

        dout_local = torch.ones_like(out, device=device, dtype=out_ref.dtype)

        dq_local, dk_local, dv_local = backward.ring_backward(
            dout_local,
            q_local,
            out,
            x,
            lse,
            w_qkv,
            scale,
            process_mesh_dict["row"],
            local_rank,
            world_size,
            chunks_to_save,
            ring_config=ring_config,
        )

        # Backward pass
        out_ref.backward(torch.ones_like(out_ref))
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

        out_core.backward(torch.ones_like(out_core))
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        ring_group = process_mesh_dict["row"]
        # dq_ref: results from calling torch.nn.SDPA
        # q_ref2: results from calling forward-backward core functionality
        # from ring_config.attn_core_forward_fn
        assert_close(
            "dq core vs sdpa", q_ref2.grad, dq_ref, atol, 1e-5, device, ring_group
        )
        assert_close(
            "dk core vs sdpa", k_ref2.grad, dk_ref, atol, 1e-5, device, ring_group
        )
        assert_close(
            "dv core vs sdpa", v_ref2.grad, dv_ref, atol, 1e-5, device, ring_group
        )

        # gather
        dq_full = gather_1d_results(dq_local, process_mesh_dict, dim=2)
        dk_full = gather_1d_results(dk_local, process_mesh_dict, dim=2)
        dv_full = gather_1d_results(dv_local, process_mesh_dict, dim=2)

        # dq_local, dq_full: results from true ring attention
        assert_close("output", out_full, out_ref, atol, 1e-5, device, ring_group)
        assert_close(
            "dq", dq_full, dq_ref.to(dq_full.dtype), atol, 1e-5, device, ring_group
        )
        assert_close(
            "dk", dk_full, dk_ref.to(dk_full.dtype), atol, 1e-5, device, ring_group
        )
        assert_close(
            "dv", dv_full, dv_ref.to(dv_full.dtype), atol, 1e-5, device, ring_group
        )

        local_slice = slice(local_rank * n_local, (local_rank + 1) * n_local)
        assert_close(
            "dq local shard",
            dq_local,
            dq_ref[:, :, local_slice].to(dq_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dk local shard",
            dk_local,
            dk_ref[:, :, local_slice].to(dk_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dv local shard",
            dv_local,
            dv_ref[:, :, local_slice].to(dv_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )


@pytest.mark.te_required
@pytest.mark.gpu_required
@pytest.mark.dist(4)
@pytest.mark.parametrize("save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL])
@pytest.mark.parametrize(
    "send_backward",
    [config.SendBackward.GATHER_DKV, config.SendBackward.SEND_DKV],
)
def test_ring_forward_backward_te(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
) -> None:
    """Ring forward/backward with the TransformerEngine inner core vs full SDPA.

    Exercises :mod:`jigsaw.modules.attention.functional.cores.te` inside the ring:
    ``fused_attn_fwd`` supplies the per-chunk ``(out, lse)`` for the online-softmax
    merge and ``fused_attn_bwd`` recomputes ``dQ/dK/dV`` from the saved *global*
    ``out`` / ``lse``.
    """
    process_mesh_dict = process_mesh_dict_4x1
    ring_config = config.RingAttentionConfig(
        device=device,
        send_in_forward=config.SendForward.X,
        send_op=config.SendOp.P2P,
        save_for_backward=config.SaveBackward.KV,
        save_scope=save_scope,
        send_backward=send_backward,
        offload=config.Offload.DISABLED,
        attn_backend=config.AttnBackend.TE,
    )
    assert ring_config.attn_core_forward_fn.__name__ == "forward"
    assert ring_config.attn_core_forward_fn.__module__.endswith("functional.cores.te")

    local_rank = dist.get_rank(group=process_mesh_dict["row"])
    world_size = dist.get_world_size(group=process_mesh_dict["row"])
    dtype = torch.bfloat16
    torch.manual_seed(2)
    b, h, n_global, d = 3, 6, world_size * 16, 16
    n_local = n_global // world_size

    w_qkv = torch.randn(3 * d * h, d, device=device, requires_grad=True, dtype=dtype) / (
        d**0.5
    )
    torch.manual_seed(local_rank)
    x = torch.randn(b, n_local, d, device=device, requires_grad=True, dtype=dtype) / (
        d**0.5
    )
    x.fill_(local_rank * 0.1)
    x = x.contiguous()

    qkv = x @ w_qkv.mT
    q_local, k_local, v_local = (
        t.contiguous()
        for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
    )
    scale = d**-0.5

    ring_forward = forward.build_ring_forward(ring_config)
    out, lse, chunks_to_save = ring_forward(
        q_local,
        k_local,
        v_local,
        x,
        w_qkv,
        scale,
        0.0,
        process_mesh_dict["row"],
        local_rank,
        world_size,
        ring_config,
    )
    out_full = gather_1d_results(out, process_mesh_dict, dim=2)

    q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(True)
    k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(True)
    v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(True)
    with torch.nn.attention.sdpa_kernel(
        [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]
    ):
        out_ref = torch.nn.functional.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, is_causal=False, dropout_p=0.0
        )

    dq_local, dk_local, dv_local = backward.ring_backward(
        torch.ones_like(out, device=device, dtype=out_ref.dtype),
        q_local,
        out,
        x,
        lse,
        w_qkv,
        scale,
        process_mesh_dict["row"],
        local_rank,
        world_size,
        chunks_to_save,
        ring_config=ring_config,
    )
    out_ref.backward(torch.ones_like(out_ref))
    dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

    atol = 2e-2
    assert torch.allclose(out_full, out_ref, atol=atol)
    for local_grad, ref_grad in (
        (dq_local, dq_ref),
        (dk_local, dk_ref),
        (dv_local, dv_ref),
    ):
        ref_slice = ref_grad[
            :, :, local_rank * n_local : (local_rank + 1) * n_local
        ].to(local_grad.dtype)
        assert torch.allclose(local_grad, ref_slice, atol=atol)


@pytest.mark.dist(8)
@pytest.mark.parametrize(
    "send_in_forward", [config.SendForward.X, config.SendForward.KV]
)
@pytest.mark.parametrize(
    "save_for_backward", [config.SaveBackward.X, config.SaveBackward.KV]
)
@pytest.mark.parametrize(
    "send_op",
    [
        config.SendOp.HYBRID_P2P,
        config.SendOp.HYBRID_ALLTOALL,
        config.SendOp.HYBRID_BROADCAST,
        config.SendOp.P2P,
        config.SendOp.ALLTOALL,
        config.SendOp.BROADCAST,
    ],
)
@pytest.mark.parametrize(
    "save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL]
)
@pytest.mark.parametrize(
    "send_backward",
    [config.SendBackward.SEND_DKV, config.SendBackward.GATHER_DKV],
)
@pytest.mark.parametrize("offloader", [config.Offload.DISABLED, config.Offload.CPU])
def test_ring_hybrid(
    device: torch.device,
    process_mesh_dict_2x4: dict,
    send_in_forward: config.SendForward,
    save_for_backward: config.SaveBackward,
    send_op: config.SendOp,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
    offloader: config.Offload,
) -> None:
    """Verify if forward-backward pass works for ring attention."""
    if (
        send_in_forward == config.SendForward.KV
        and save_for_backward == config.SaveBackward.X
    ):
        pytest.xfail("not yet supported")

        process_mesh_dict = process_mesh_dict_2x4

        ring_config = config.RingAttentionConfig(
            device=device,
            send_in_forward=send_in_forward,
            send_op=send_op,
            save_for_backward=save_for_backward,
            save_scope=save_scope,
            send_backward=send_backward,
            offload=offloader,
        )

        ring_forward = forward.build_ring_forward(ring_config)
        local_rank = dist.get_rank(group=process_mesh_dict["model"])
        world_size = dist.get_world_size(group=process_mesh_dict["model"])

        dtype = torch.bfloat16 if dist.get_backend() == "nccl" else torch.float32
        torch.manual_seed(2)  # same seed for weight initialization
        b, h, n_global, d = 3, 6, world_size * 16, 16
        n_local = n_global // world_size

        w_qkv = torch.randn(
            3 * d * h, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        w_qkv.fill_(0.01)
        torch.manual_seed(local_rank)  # different seed for input data generation
        x = torch.randn(
            b, n_local, d, device=device, requires_grad=True, dtype=dtype
        ) / (d**0.5)
        x = x.contiguous()
        x = x.fill_(0.1 * local_rank)

        qkv = x @ w_qkv.mT
        # QKV shape: [B, Nlocal, 3*d*h]
        # We need QKV to have shape: []
        q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(
            2, 0, 3, 1, 4
        )

        q_local, k_local, v_local = (
            q_local.contiguous(),
            k_local.contiguous(),
            v_local.contiguous(),
        )
        scale = q_local.shape[-1] ** -0.5

        # just test helper function forward
        ring_forward = forward.build_ring_forward(ring_config)
        out, lse, chunks_to_save = ring_forward(
            q_local,
            k_local,
            v_local,
            x,
            w_qkv,
            scale,
            0.0,
            group=process_mesh_dict["model"],
            local_rank=local_rank,
            ring_group_size=world_size,
            ring_config=ring_config,
        )

        out_full = gather_1d_results(out, process_mesh_dict, dim=2)
        x_full = gather_1d_results(x, process_mesh_dict, dim=1)
        qkv = x_full @ w_qkv.mT
        q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )
        v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(
            True
        )

        # Reference calculation
        with torch.nn.attention.sdpa_kernel(
            [
                torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                torch.nn.attention.SDPBackend.MATH,
            ]
        ):
            out_ref = torch.nn.functional.scaled_dot_product_attention(
                q_ref, k_ref, v_ref, is_causal=False, dropout_p=0.0
            )

        q_ref2, k_ref2, v_ref2 = (
            q_ref.detach().clone(),
            k_ref.detach().clone(),
            v_ref.detach().clone(),
        )
        q_ref2 = q_ref2.requires_grad_(True)
        k_ref2 = k_ref2.requires_grad_(True)
        v_ref2 = v_ref2.requires_grad_(True)

        out_core = ring_config.attn_core_forward_fn(
            q_ref2, k_ref2, v_ref2, dropout_p=0.0, scale=scale
        )[0]
        out_core = out_core.requires_grad_(True)

        dout_local = torch.ones_like(out, device=device, dtype=out_ref.dtype)
        dq_local, dk_local, dv_local = backward.ring_backward(
            dout_local,
            q_local,
            out,
            x,
            lse,
            w_qkv,
            scale,
            process_mesh_dict["model"],
            local_rank,
            world_size,
            chunks_to_save,
            ring_config=ring_config,
        )

        # Backward pass
        out_ref.backward(torch.ones_like(out_ref))
        dq_ref, dk_ref, dv_ref = q_ref.grad, k_ref.grad, v_ref.grad

        out_core.backward(torch.ones_like(out_core))
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        ring_group = process_mesh_dict["model"]
        # dq_ref: results from calling torch.nn.SDPA
        # q_ref2: results from calling forward-backward core functionality
        # from ring_config.attn_core_forward_fn

        assert_close(
            "dq core vs sdpa", q_ref2.grad, dq_ref, atol, 1e-5, device, ring_group
        )
        assert_close(
            "dk core vs sdpa", k_ref2.grad, dk_ref, atol, 1e-5, device, ring_group
        )
        assert_close(
            "dv core vs sdpa", v_ref2.grad, dv_ref, atol, 1e-5, device, ring_group
        )

        # gather
        dq_full = gather_1d_results(dq_local, process_mesh_dict, dim=2)
        dk_full = gather_1d_results(dk_local, process_mesh_dict, dim=2)
        dv_full = gather_1d_results(dv_local, process_mesh_dict, dim=2)

        # dq_local, dq_full: results from true ring attention
        assert_close("output", out_full, out_ref, atol, 1e-5, device, ring_group)
        assert_close(
            "dq sum",
            dq_full.sum(),
            dq_ref.to(dq_full.dtype).sum(),
            1e-8,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dq", dq_full, dq_ref.to(dq_full.dtype), atol, 1e-5, device, ring_group
        )
        # assert torch.allclose(dk_full.sum(), dk_ref.to(dk_full.dtype).sum())
        assert_close(
            "dk", dk_full, dk_ref.to(dk_full.dtype), atol, 1e-5, device, ring_group
        )
        assert_close(
            "dv", dv_full, dv_ref.to(dv_full.dtype), atol, 1e-5, device, ring_group
        )

        local_slice = slice(local_rank * n_local, (local_rank + 1) * n_local)
        assert_close(
            "dq local shard",
            dq_local,
            dq_ref[:, :, local_slice].to(dq_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dk local shard",
            dk_local,
            dk_ref[:, :, local_slice].to(dk_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
        assert_close(
            "dv local shard",
            dv_local,
            dv_ref[:, :, local_slice].to(dv_local.dtype),
            atol,
            1e-5,
            device,
            ring_group,
        )
