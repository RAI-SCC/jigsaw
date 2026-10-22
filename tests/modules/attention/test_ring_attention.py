import itertools

import pytest
import torch
import torch.distributed as dist

import jigsaw.modules.attention._config as config
from jigsaw.gather import (
    gather_1d_results,
)
from jigsaw.modules.attention._functional import backward, forward, ring
from jigsaw.modules.attention._functional.cores import te as te_cores
from tests.parity import assert_close

pytestmark = [pytest.mark.dist_required]


class TestHybridTopology:
    @staticmethod
    def _assert_subgroup_peers_agree(
        topo: ring.HybridTopology, ring_global_ranks: list[int]
    ) -> None:
        """The sub-group peer accessors must name the same ranks as the ring ones.

        Every hybrid hop addresses its peer with ``intranode_peer`` /
        ``internode_peer`` inside the matching sub-group; the chunk-index
        bookkeeping still reasons in ring positions via ``intra_peer`` /
        ``inter_peer``. Resolving both through the sub-group's global rank list
        pins the two spellings to the same physical rank.
        """
        intra = dist.get_process_group_ranks(topo.intranode_group)
        inter = dist.get_process_group_ranks(topo.internode_group)
        for off in range(-2, 3):
            assert (
                intra[topo.intranode_peer(off)]
                == ring_global_ranks[topo.intra_peer(off)]
            )
            assert (
                inter[topo.internode_peer(off)]
                == ring_global_ranks[topo.inter_peer(off)]
            )

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

        ring._HYBRID_TOPOLOGY_CACHE.pop((group, gpus_per_node), None)
        topo = ring.get_hybrid_topology(group, gpus_per_node)

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

        self._assert_subgroup_peers_agree(topo, dist.get_process_group_ranks(group))

        # Cached: same object on the second call.
        assert ring.get_hybrid_topology(group, gpus_per_node) is topo

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

        ring._HYBRID_TOPOLOGY_CACHE.pop((group, gpus_per_node), None)
        topo = ring.get_hybrid_topology(group, gpus_per_node)

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
        self._assert_subgroup_peers_agree(topo, ring_global_ranks)

        # Every member of a sub-group lands on the same underlying group.
        assert ring.get_hybrid_topology(group, gpus_per_node) is topo

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
            ring.get_hybrid_topology(group, 0)
        with pytest.raises(ValueError, match="divisible by gpus_per_node"):
            ring.get_hybrid_topology(group, 3)


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
        if offloader == config.Offload.CPU and device.type != "cuda":
            pytest.skip("Offload.CPU relies on CUDA streams and needs a cuda device")

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

        ring_forward = forward.ring_forward
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
        ring_forward = forward.ring_forward
        out, lse, chunks_to_save, base_seed, rng_states = ring_forward(
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
@pytest.mark.parametrize(
    "save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL]
)
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

    Exercises :mod:`jigsaw.modules.attention._functional.cores.te` inside the ring:
    whichever arm this GPU selects supplies the per-chunk ``(out, lse)`` for the
    online-softmax merge, and its backward recomputes ``dQ/dK/dV`` from the saved
    *global* ``out`` / ``lse``. Set ``JIGSAW_TE_CORE`` to exercise the arm this
    architecture does not pick.
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
    # `te` resolves to one of two arms by GPU architecture (cuDNN on Hopper+,
    # flash_attn on Ampere), so pin it to whichever `select_core` names here.
    expected = te_cores.select_core(device).__name__
    assert ring_config.attn_core_forward_fn.__module__ == expected
    assert expected.endswith((".te.cudnn", ".te.flash"))

    local_rank = dist.get_rank(group=process_mesh_dict["row"])
    world_size = dist.get_world_size(group=process_mesh_dict["row"])
    dtype = torch.bfloat16
    torch.manual_seed(2)
    b, h, n_global, d = 3, 6, world_size * 16, 16
    n_local = n_global // world_size

    w_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=dtype
    ) / (d**0.5)
    torch.manual_seed(local_rank)
    x = torch.randn(b, n_local, d, device=device, requires_grad=True, dtype=dtype) / (
        d**0.5
    )
    x = x.contiguous()

    qkv = x @ w_qkv.mT
    # The TE ring core is bshd-native (``core_layout == "bshd"``), so the ring
    # takes / returns ``(b, s, h, d)``; the SDPA reference below stays bhsd and
    # the two are bridged with a transpose.
    q_local, k_local, v_local = (
        t.contiguous() for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 1, 3, 4)
    )
    scale = d**-0.5

    ring_forward = forward.ring_forward
    out, lse, chunks_to_save, base_seed, rng_states = ring_forward(
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
    out_full = gather_1d_results(out, process_mesh_dict, dim=1).transpose(1, 2)

    def _ref(t: torch.Tensor) -> torch.Tensor:
        return (
            gather_1d_results(t, process_mesh_dict, dim=1)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

    q_ref, k_ref, v_ref = _ref(q_local), _ref(k_local), _ref(v_local)
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
        # ring grad is bshd -> bhsd, then slice this rank's chunk on the seq axis
        local_bhsd = local_grad.transpose(1, 2)
        ref_slice = ref_grad[
            :, :, local_rank * n_local : (local_rank + 1) * n_local
        ].to(local_bhsd.dtype)
        assert torch.allclose(local_bhsd, ref_slice, atol=atol)


# Shared parametrization for the ``test_ring_hybrid*`` variants. The full product
# of the six options is 192 cases per topology; each costs a ring of collectives,
# so the default run takes a pairwise cover (every pair of option values appears
# in some case) and the rest is ``slow`` (``--runslow``, before a release). The
# cover is built for each offload mode separately: ``Offload.CPU`` skips without
# CUDA, so a CPU run still covers every pair of the other five options.
_RING_HYBRID_OPTIONS = {
    "send_in_forward": [config.SendForward.X, config.SendForward.KV],
    "save_for_backward": [config.SaveBackward.X, config.SaveBackward.KV],
    "send_op": [
        config.SendOp.HYBRID_P2P,
        config.SendOp.HYBRID_ALLTOALL,
        config.SendOp.HYBRID_BROADCAST,
        config.SendOp.P2P,
        config.SendOp.ALLTOALL,
        config.SendOp.BROADCAST,
    ],
    "save_scope": [config.SaveScope.LOCAL, config.SaveScope.GLOBAL],
    "send_backward": [config.SendBackward.SEND_DKV, config.SendBackward.GATHER_DKV],
}
_OFFLOADERS = [config.Offload.DISABLED, config.Offload.CPU]


def _ring_hybrid_unsupported(combo: tuple) -> bool:
    """``SendForward.KV`` with ``SaveBackward.X``: xfails in ``_check_ring_hybrid``."""
    return combo[0] == config.SendForward.KV and combo[1] == config.SaveBackward.X


def _pairwise_cover(combos: list[tuple]) -> set[tuple]:
    """Greedy (deterministic) subset of ``combos`` covering every value pair.

    Pairs are taken over the supported combos only; one unsupported combo is
    added so its xfail stays visible in the default run.
    """

    def pairs(c: tuple) -> set[tuple]:
        return {(i, c[i], j, c[j]) for i, j in itertools.combinations(range(len(c)), 2)}

    supported = [c for c in combos if not _ring_hybrid_unsupported(c)]
    uncovered = set().union(*map(pairs, supported))
    chosen = set()
    while uncovered:
        best = max(supported, key=lambda c: len(pairs(c) & uncovered))
        chosen.add(best)
        uncovered -= pairs(best)
    chosen.add(next(c for c in combos if _ring_hybrid_unsupported(c)))
    return chosen


def _ring_hybrid_cases() -> list:
    combos = list(itertools.product(*_RING_HYBRID_OPTIONS.values()))
    default = _pairwise_cover(combos)
    return [
        pytest.param(
            *combo,
            offloader,
            id="-".join(str(v) for v in (*combo, offloader)),
            marks=() if combo in default else pytest.mark.slow,
        )
        for offloader in _OFFLOADERS
        for combo in combos
    ]


def _apply_ring_hybrid_parametrize(fn):
    """Apply the shared ``test_ring_hybrid*`` parametrization to ``fn``."""
    return pytest.mark.parametrize(
        (*_RING_HYBRID_OPTIONS, "offloader"), _ring_hybrid_cases()
    )(fn)


def _check_ring_hybrid(
    device: torch.device,
    process_mesh_dict: dict,
    send_in_forward: config.SendForward,
    save_for_backward: config.SaveBackward,
    send_op: config.SendOp,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
    offloader: config.Offload,
    gpus_per_node: int,
) -> None:
    """Forward+backward parity check for one hybrid ring topology.

    Shared body for :func:`test_ring_hybrid` (8 ranks, ``2 nodes x 4 GPUs``)
    and :func:`test_ring_hybrid_single_node` (4 ranks, ``2 nodes x 2 GPUs``).
    The hybrid node split is driven purely by ``gpus_per_node``, so both keep
    ``n_nodes == 2`` and every ``HYBRID_*`` send op exercises a real intra-node
    and a real inter-node sub-group.
    """
    if (
        send_in_forward == config.SendForward.KV
        and save_for_backward == config.SaveBackward.X
    ):
        pytest.xfail("not yet supported")
    if offloader == config.Offload.CPU and device.type != "cuda":
        pytest.skip("Offload.CPU relies on CUDA streams and needs a cuda device")

    ring_config = config.RingAttentionConfig(
        device=device,
        send_in_forward=send_in_forward,
        send_op=send_op,
        save_for_backward=save_for_backward,
        save_scope=save_scope,
        send_backward=send_backward,
        offload=offloader,
        gpus_per_node=gpus_per_node,
    )

    ring_forward = forward.ring_forward
    local_rank = dist.get_rank(group=process_mesh_dict["model"])
    world_size = dist.get_world_size(group=process_mesh_dict["model"])

    dtype = torch.bfloat16 if dist.get_backend() == "nccl" else torch.float32
    torch.manual_seed(2)  # same seed for weight initialization
    b, h, n_global, d = 3, 6, world_size * 16, 16
    n_local = n_global // world_size

    w_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=dtype
    ) / (d**0.5)
    torch.manual_seed(local_rank)  # different seed for input data generation
    x = torch.randn(b, n_local, d, device=device, requires_grad=True, dtype=dtype) / (
        d**0.5
    )
    x = x.contiguous()

    qkv = x @ w_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    # We need QKV to have shape: []
    q_local, k_local, v_local = qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)

    q_local, k_local, v_local = (
        q_local.contiguous(),
        k_local.contiguous(),
        v_local.contiguous(),
    )
    scale = q_local.shape[-1] ** -0.5

    # just test helper function forward
    ring_forward = forward.ring_forward
    out, lse, chunks_to_save, base_seed, rng_states = ring_forward(
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
    q_ref = gather_1d_results(q_local, process_mesh_dict, dim=2).requires_grad_(True)
    k_ref = gather_1d_results(k_local, process_mesh_dict, dim=2).requires_grad_(True)
    v_ref = gather_1d_results(v_local, process_mesh_dict, dim=2).requires_grad_(True)

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

    assert_close("dq core vs sdpa", q_ref2.grad, dq_ref, atol, 1e-5, device, ring_group)
    assert_close("dk core vs sdpa", k_ref2.grad, dk_ref, atol, 1e-5, device, ring_group)
    assert_close("dv core vs sdpa", v_ref2.grad, dv_ref, atol, 1e-5, device, ring_group)

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


@pytest.mark.dist(8)
@_apply_ring_hybrid_parametrize
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
    """Verify forward-backward parity for the hybrid ring on 8 ranks.

    The ring spans the full ``2 x 4`` model mesh and splits into
    ``2 nodes x 4 GPUs`` (``gpus_per_node=4``).
    """
    _check_ring_hybrid(
        device,
        process_mesh_dict_2x4,
        send_in_forward,
        save_for_backward,
        send_op,
        save_scope,
        send_backward,
        offloader,
        gpus_per_node=4,
    )


@pytest.mark.dist(6)
@_apply_ring_hybrid_parametrize
def test_ring_hybrid_three_nodes(
    device: torch.device,
    process_mesh_dict_2x3: dict,
    send_in_forward: config.SendForward,
    save_for_backward: config.SaveBackward,
    send_op: config.SendOp,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
    offloader: config.Offload,
) -> None:
    """Verify forward-backward parity for the hybrid ring on 6 ranks.

    Runs the ring as ``3 nodes x 2 GPUs`` (``gpus_per_node=2``). Two nodes are not
    enough to exercise the inter-node relay (a chunk hopped on to a further node),
    which only happens while ``node_step < n_nodes - 1``.
    """
    _check_ring_hybrid(
        device,
        process_mesh_dict_2x3,
        send_in_forward,
        save_for_backward,
        send_op,
        save_scope,
        send_backward,
        offloader,
        gpus_per_node=2,
    )


@pytest.mark.dist(8)
@_apply_ring_hybrid_parametrize
def test_ring_hybrid_four_nodes(
    device: torch.device,
    process_mesh_dict_2x4: dict,
    send_in_forward: config.SendForward,
    save_for_backward: config.SaveBackward,
    send_op: config.SendOp,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
    offloader: config.Offload,
) -> None:
    """Verify forward-backward parity for the hybrid ring with >2 inter-node hops.

    Same 8-rank mesh as :func:`test_ring_hybrid`, but ``gpus_per_node=2`` ->
    ``4 nodes x 2 GPUs``, so the inter-node ring takes 3 hops instead of 1. At
    ``n_nodes == 2`` (every other hybrid test) ``internode_peer(1)`` and
    ``internode_peer(-1)`` name the same rank, so a provider that keeps
    re-sending its *original* chunk every round instead of relaying the chunk
    it just received is indistinguishable from a correct one -- this is the
    smallest topology where the two diverge.
    """
    _check_ring_hybrid(
        device,
        process_mesh_dict_2x4,
        send_in_forward,
        save_for_backward,
        send_op,
        save_scope,
        send_backward,
        offloader,
        gpus_per_node=2,
    )


@pytest.mark.dist(4)
@_apply_ring_hybrid_parametrize
def test_ring_hybrid_single_node(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    send_in_forward: config.SendForward,
    save_for_backward: config.SaveBackward,
    send_op: config.SendOp,
    save_scope: config.SaveScope,
    send_backward: config.SendBackward,
    offloader: config.Offload,
) -> None:
    """Verify forward-backward parity for the hybrid ring on 4 ranks.

    Runs the 4-rank ring as ``2 nodes x 2 GPUs`` (``gpus_per_node=2``) so every
    hybrid send op still exercises both its intra-node (2 members) and
    inter-node (2 members) paths while fitting on a single 2-GPU node.
    """
    _check_ring_hybrid(
        device,
        process_mesh_dict_4x1,
        send_in_forward,
        save_for_backward,
        send_op,
        save_scope,
        send_backward,
        offloader,
        gpus_per_node=2,
    )


@pytest.mark.dist(4)
def test_ring_autograd_function_matches_full_attention(
    device: torch.device, process_mesh_dict_4x1: dict
) -> None:
    """``_RingAttention`` gives the gradients of full attention.

    Goes through the autograd wrapper (``setup_context`` saves the log-sum-exp for the
    backward) with ``heads != local sequence length``; a transposed saved ``lse``
    crashes there and is only silently wrong when the two happen to be equal.
    """
    mesh = process_mesh_dict_4x1
    group = mesh["row"]
    rank, world = group.rank(), group.size()
    b, h, n_local, d = 2, 3, 8, 8
    assert h != n_local

    torch.manual_seed(0)  # identical full tensors on every rank
    q_full, k_full, v_full, g_full = (
        torch.randn(b, h, world * n_local, d, device=device) for _ in range(4)
    )
    shard = slice(rank * n_local, (rank + 1) * n_local)
    q, k, v = (
        t[:, :, shard].clone().contiguous().requires_grad_(True)
        for t in (q_full, k_full, v_full)
    )
    q_ref, k_ref, v_ref = (
        t.clone().requires_grad_(True) for t in (q_full, k_full, v_full)
    )
    out_ref = torch.nn.functional.scaled_dot_product_attention(q_ref, k_ref, v_ref)
    out_ref.backward(g_full)

    # ``KV`` sends the keys/values themselves, so ``x`` and ``w_qkv`` are not read.
    x = torch.randn(b, n_local, d, device=device)
    w_qkv = torch.randn(3 * h * d, d, device=device)
    ring_config = config.RingAttentionConfig(
        device=device, gpus_per_node=world, send_in_forward=config.SendForward.KV
    )
    out = ring._RingAttention.apply(
        q, k, v, x, w_qkv, d**-0.5, 0.0, group, rank, ring_config
    )[0]
    out.backward(g_full[:, :, shard].contiguous())

    # The CUDA core runs flash attention in float16; a transposed
    # ``lse`` is off by O(1), so this still catches it.
    tol = 5e-3 if device.type == "cuda" else 1e-4
    for name, got, ref in (
        ("out", out, out_ref),
        ("dq", q.grad, q_ref.grad),
        ("dk", k.grad, k_ref.grad),
        ("dv", v.grad, v_ref.grad),
    ):
        assert got is not None and ref is not None
        assert_close(name, got, ref[:, :, shard], tol, tol, device, mesh["model"])


# --------------------------------------------------------------------------- #
# comm-stream overlap (RingAttentionConfig.overlap_comm)
# --------------------------------------------------------------------------- #
@pytest.mark.gpu_required
@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "send_in_forward", [config.SendForward.KV, config.SendForward.X]
)
@pytest.mark.parametrize(
    "save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL]
)
def test_ring_overlap_comm_parity(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    send_in_forward: config.SendForward,
    save_scope: config.SaveScope,
) -> None:
    """``overlap_comm`` is a pure scheduling change: same out / dq / dk / dv.

    Runs the plain ``SendOp.P2P`` ring forward+backward with the comm-stream
    overlap off and then on (dedicated CUDA stream + double-buffered K/V
    circulation, hoisted dK/dV relay) and asserts the results are numerically
    identical -- the ops and their accumulation order are unchanged, only which
    stream they run on. ``send_backward`` is ``SEND_DKV`` so ``save_scope=LOCAL``
    exercises the ``_DkvRelay`` async path and ``GLOBAL`` the all-reduce path
    (overlap inert there, so it doubles as a no-regression check).
    """
    group = process_mesh_dict_4x1["row"]
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    dtype = torch.bfloat16
    torch.manual_seed(2)
    b, h, n_global, d = 3, 6, world_size * 16, 16
    n_local = n_global // world_size

    w_qkv = torch.randn(3 * d * h, d, device=device, dtype=dtype) / (d**0.5)
    torch.manual_seed(local_rank)
    x = (torch.randn(b, n_local, d, device=device, dtype=dtype) / (d**0.5)).contiguous()
    qkv = x @ w_qkv.mT
    q, k, v = (
        t.contiguous() for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
    )
    scale = d**-0.5
    dout = torch.ones(b, h, n_local, d, device=device, dtype=dtype)

    def run(overlap: bool):
        cfg = config.RingAttentionConfig(
            device=device,
            send_in_forward=send_in_forward,
            send_op=config.SendOp.P2P,
            save_for_backward=config.SaveBackward.KV,
            save_scope=save_scope,
            send_backward=config.SendBackward.SEND_DKV,
            overlap_comm=overlap,
        )
        assert cfg.overlap_active == overlap  # CUDA build => stream created
        out, lse, chunks, seed, rng = forward.ring_forward(
            q, k, v, x, w_qkv, scale, 0.0, group, local_rank, world_size, cfg
        )
        dq, dk, dv = backward.ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            scale,
            group,
            local_rank,
            world_size,
            chunks,
            ring_config=cfg,
            dropout_p=0.0,
            base_seed=seed,
            rng_states=rng,
        )
        torch.cuda.synchronize()
        return out, dq, dk, dv

    ref = run(False)
    got = run(True)
    for name, a, c in zip(("out", "dq", "dk", "dv"), ref, got, strict=True):
        assert torch.allclose(a, c, atol=1e-3, rtol=0), (
            f"{name}: overlap changed the result "
            f"(max |Δ| = {(a.float() - c.float()).abs().max().item():.3e})"
        )


@pytest.mark.gpu_required
@pytest.mark.dist(4)
@pytest.mark.parametrize(
    "send_in_forward", [config.SendForward.KV, config.SendForward.X]
)
@pytest.mark.parametrize(
    "save_scope", [config.SaveScope.LOCAL, config.SaveScope.GLOBAL]
)
@pytest.mark.parametrize(
    "send_op",
    [
        config.SendOp.HYBRID_P2P,
        config.SendOp.HYBRID_ALLTOALL,
        config.SendOp.HYBRID_BROADCAST,
    ],
)
def test_ring_overlap_comm_parity_hybrid(
    device: torch.device,
    process_mesh_dict_4x1: dict,
    send_in_forward: config.SendForward,
    save_scope: config.SaveScope,
    send_op: config.SendOp,
) -> None:
    """``overlap_comm`` parity for the hybrid send ops.

    The hybrid twin of :func:`test_ring_overlap_comm_parity`: runs the hybrid
    ring (``2 nodes x 2 GPUs``, ``gpus_per_node=2``) forward+backward with the
    comm-stream overlap off and then on and asserts identical results. Covers
    the intra-/inter-node K/V hop overlap
    (``_HybridP2PForwardChunkProviderOverlap`` /
    ``_HybridBroadcastForwardChunkProviderOverlap`` and their backward twins)
    for all three hybrid send ops, and -- for ``HYBRID_P2P`` /
    ``HYBRID_ALLTOALL`` with ``save_scope=LOCAL`` -- the hybrid-routed dK/dV
    relay (``_DkvRelay`` via ``_hybrid_dkv_relay_peers``). ``HYBRID_BROADCAST``
    at ``LOCAL`` scope exercises its (unoverlapped) dK/dV all-reduce path
    alongside the overlapped K/V providers, doubling as a no-regression check
    for that combination.
    """
    group = process_mesh_dict_4x1["model"]
    local_rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    dtype = torch.bfloat16
    torch.manual_seed(2)
    b, h, n_global, d = 3, 6, world_size * 16, 16
    n_local = n_global // world_size

    w_qkv = torch.randn(3 * d * h, d, device=device, dtype=dtype) / (d**0.5)
    torch.manual_seed(local_rank)
    x = (torch.randn(b, n_local, d, device=device, dtype=dtype) / (d**0.5)).contiguous()
    qkv = x @ w_qkv.mT
    q, k, v = (
        t.contiguous() for t in qkv.reshape(b, n_local, 3, h, d).permute(2, 0, 3, 1, 4)
    )
    scale = d**-0.5
    dout = torch.ones(b, h, n_local, d, device=device, dtype=dtype)

    def run(overlap: bool):
        cfg = config.RingAttentionConfig(
            device=device,
            send_in_forward=send_in_forward,
            send_op=send_op,
            save_for_backward=config.SaveBackward.KV,
            save_scope=save_scope,
            send_backward=config.SendBackward.SEND_DKV,
            gpus_per_node=2,
            overlap_comm=overlap,
        )
        assert cfg.overlap_active == overlap  # CUDA build => stream created
        out, lse, chunks, seed, rng = forward.ring_forward(
            q, k, v, x, w_qkv, scale, 0.0, group, local_rank, world_size, cfg
        )
        dq, dk, dv = backward.ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            scale,
            group,
            local_rank,
            world_size,
            chunks,
            ring_config=cfg,
            dropout_p=0.0,
            base_seed=seed,
            rng_states=rng,
        )
        torch.cuda.synchronize()
        return out, dq, dk, dv

    ref = run(False)
    got = run(True)
    for name, a, c in zip(("out", "dq", "dk", "dv"), ref, got, strict=True):
        assert torch.allclose(a, c, atol=1e-3, rtol=0), (
            f"{name}: overlap changed the result "
            f"(max |Δ| = {(a.float() - c.float()).abs().max().item():.3e})"
        )
