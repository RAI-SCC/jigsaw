"""Ring attention forward-backward implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.distributed as dist

from ..config import RingAttentionConfig


def _compute_kv(
    xj: torch.Tensor, w_qkv: torch.Tensor, b: int, n_local: int, h: int, d: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute key and value tensors from an input chunk.

    Parameters
    ----------
    xj : torch.Tensor
        Input sequence chunk.
    w_qkv : torch.Tensor
        Fused query, key, and value projection weights.
    b : int
        Batch size.
    n_local : int
        Sequence length of the local chunk.
    h : int
        Number of attention heads.
    d : int
        Embedding dimension per head.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The fused key-value tensor and its key and value views.
    """
    kv = (
        (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    return kv, kv[0], kv[1]


# --------------------------------------------------------------------------- #
# hybrid (intra-node + inter-node) ring topology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HybridTopology:
    """Node/rank layout of a 1-D ring split into equal-sized nodes.

    The ring's ``world_size`` ranks are cut into ``n_nodes`` contiguous nodes of
    ``gpus_per_node`` ranks each. Ring position ``p`` (``0 <= p < world_size``)
    owns global sequence chunk ``p`` and lives at ``(p // gpus_per_node,
    p % gpus_per_node)``. Every peer lookup a hybrid ring needs -- the next rank
    along the intra-node ring, the next along the inter-node ring, or the rank
    at an arbitrary ``(node, local)`` -- is a table lookup here, so callers
    never open-code ``// gpus_per_node`` / ``% gpus_per_node`` arithmetic.

    ``intranode_group`` / ``internode_group`` are the sub-groups containing this
    rank, provided for collectives (e.g. broadcasts); the P2P ring paths address
    peers by ring position over the full group instead.
    """

    n_nodes: int
    gpus_per_node: int
    node_idx: int
    local_idx: int
    intranode_group: dist.ProcessGroup
    internode_group: dist.ProcessGroup
    _grid: tuple[tuple[int, ...], ...]

    @property
    def ring_rank(self) -> int:
        """Position of this rank in the ring (``node_idx * gpn + local_idx``)."""
        return self.node_idx * self.gpus_per_node + self.local_idx

    def rank_at(self, node: int, local: int) -> int:
        """Ring position at ``(node, local)``, both taken modulo their extent."""
        return self._grid[node % self.n_nodes][local % self.gpus_per_node]

    def intra_peer(self, offset: int) -> int:
        """Ring position ``offset`` steps along this rank's intra-node ring."""
        return self.rank_at(self.node_idx, self.local_idx + offset)

    def inter_peer(self, offset: int) -> int:
        """Ring position ``offset`` steps along this rank's inter-node ring."""
        return self.rank_at(self.node_idx + offset, self.local_idx)


_HYBRID_TOPOLOGY_CACHE: dict[tuple[dist.ProcessGroup, int], HybridTopology] = {}


def get_hybrid_topology(group: dist.ProcessGroup, gpus_per_node: int) -> HybridTopology:
    """Build (and cache) the :class:`HybridTopology` for ``group``.

    Sub-group creation is a world-wide collective:
    :func:`torch.distributed.new_subgroups_by_enumeration` issues one
    ``new_group`` per sub-group and every rank must enter every call in the same
    order. ``group`` must therefore span the whole world, which is also what the
    hybrid P2P paths assume when they address peers by ring position (ring
    position ``==`` global rank). The result is cached per
    ``(group, gpus_per_node)``; the group is used as the key directly so a freed
    process group can never alias a live one (as ``id(group)`` could).

    Parameters
    ----------
    group : dist.ProcessGroup
        The ring's process group; must contain every rank in the default group.
    gpus_per_node : int
        Number of consecutive ring ranks that share a node.

    Returns
    -------
    HybridTopology

    Raises
    ------
    ValueError
        If ``gpus_per_node`` is not positive, ``group`` does not span the whole
        world, or the world size is not divisible by ``gpus_per_node``.
    """
    key = (group, gpus_per_node)
    cached = _HYBRID_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    if gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be positive")
    world_size = dist.get_world_size(group)
    if world_size != dist.get_world_size():
        raise ValueError("get_hybrid_topology requires a group spanning the whole world")
    if world_size % gpus_per_node:
        raise ValueError("ring size must be divisible by gpus_per_node")
    n_nodes = world_size // gpus_per_node

    # Ring position p lives at (p // gpus_per_node, p % gpus_per_node); with the
    # whole-world assumption ring position == global rank, so the two partitions
    # below are expressed directly in global ranks.
    grid = tuple(
        tuple(node * gpus_per_node + local for local in range(gpus_per_node))
        for node in range(n_nodes)
    )
    # Contiguous chunk -> intra-node ring; same local offset across nodes ->
    # inter-node ring. Each is a full partition of the world, so
    # `new_subgroups_by_enumeration` builds every sub-group in one fixed order
    # and hands back the one holding this rank.
    intranode_group, _ = dist.new_subgroups_by_enumeration(
        [list(row) for row in grid]
    )
    internode_group, _ = dist.new_subgroups_by_enumeration(
        [[grid[node][local] for node in range(n_nodes)] for local in range(gpus_per_node)]
    )

    node_idx, local_idx = divmod(dist.get_rank(group), gpus_per_node)
    topo = HybridTopology(
        n_nodes=n_nodes,
        gpus_per_node=gpus_per_node,
        node_idx=node_idx,
        local_idx=local_idx,
        intranode_group=intranode_group,
        internode_group=internode_group,
        _grid=grid,
    )
    _HYBRID_TOPOLOGY_CACHE[key] = topo
    return topo


class RingCtx(Protocol):
    """Protocol for types stored within ctx."""

    empty: torch.Tensor
    scale: float
    local_rank: int
    group: dist.ProcessGroup
    world_size: int
    config: RingAttentionConfig
    saved_tensors: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple
    ]


# --------------------------------------------------------------------------- #
# autograd.Function glue
# --------------------------------------------------------------------------- #
class RingAttentionFunction(torch.autograd.Function):
    """Ring Attention autograd function."""

    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        w_qkv: torch.Tensor,
        scale: float,
        dropout_p: float,
        group: dist.ProcessGroup,
        empty: torch.Tensor,
        local_rank: int,
        config: RingAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """Run the ring-attention forward pass.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Local query, key, and value tensors.
        x : torch.Tensor
            Local input sequence chunk.
        w_qkv : torch.Tensor
            Fused query, key, and value projection weights.
        scale : float
            Attention scaling factor.
        dropout_p : float
            Attention dropout probability.
        group : dist.ProcessGroup
            Distributed process group used for ring communication.
        empty : torch.Tensor
            Empty tensor used by communication helpers.
        local_rank : int
            Rank within the process group.
        config : RingAttentionConfig
            Ring-attention configuration.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple]
            Attention output, log-sum-exp values, and saved chunks.
        """
        world_size = dist.get_world_size(group)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)
        from .forward import build_ring_forward

        ring_forward_fn = build_ring_forward(config)
        out, lse, chunks_to_save = ring_forward_fn(
            q,
            k,
            v,
            x,
            w_qkv,
            scale,
            dropout_p,
            group,
            empty,
            local_rank,
            world_size,
            config,
        )
        return out, lse, chunks_to_save

    @staticmethod
    def setup_context(
        ctx: torch.autograd.function.FunctionCtx,
        inputs: Any,
        output: Any,
    ) -> None:
        """Save forward results and metadata for the backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context receiving saved tensors and metadata.
        inputs : Any
            Inputs passed to :meth:`forward`.
        output : Any
            Values returned by :meth:`forward`.
        """
        q, k, v, x, w_qkv, scale, dropout_p, group, empty, local_rank, config = inputs
        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)
        ctx.save_for_backward(
            q,
            out,
            x,
            lse.mT,
            w_qkv,
            *chunks_to_save,  # *[t for pair in chunks_to_save for t in pair]
        )
        c = cast(RingCtx, ctx)
        c.empty = empty
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.world_size = dist.get_world_size(
            group
        )  # This is recomputed in forward and in setup context
        c.config = config

    @staticmethod
    def backward(
        ctx: RingCtx,
        dout: torch.Tensor,
        grad_l: torch.Tensor,
        grad_kv_chunks: list,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        """Compute gradients for the ring-attention forward pass.

        Parameters
        ----------
        ctx : RingCtx
            Autograd context containing saved tensors and configuration.
        dout : torch.Tensor
            Gradient of the attention output.
        grad_l : torch.Tensor
            Gradient placeholder for the non-differentiable log-sum-exp output.
        grad_kv_chunks : list
            Gradient placeholders for saved non-differentiable chunks.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Gradients for differentiable inputs followed by ``None`` for
            non-differentiable inputs.
        """
        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        from .backward import ring_backward

        dq, dk, dv = ring_backward(
            dout,
            q,
            out,
            x,
            lse,
            w_qkv,
            ctx.scale,
            ctx.group,
            ctx.local_rank,
            ctx.world_size,
            cast(tuple[torch.Tensor, ...], chunks_to_save),
            ring_config=ctx.config,
            empty=ctx.empty,
        )

        return dq, dk, dv, None, None, None, None, None, None, None, None
