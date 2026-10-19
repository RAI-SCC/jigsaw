"""Ring attention autograd function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from .._config import RingAttentionConfig

_SEED_MODULUS = 2**31 - 1


def _combine_dropout_seed(
    base_seed: int, local_rank: int, global_chunk_idx: int
) -> int:
    """Derive a dropout seed for one (rank, chunk) pair from a base seed.

    Parameters
    ----------
    base_seed : int
        Base seed of the ring-attention call.
    local_rank : int
        Rank within the ring.
    global_chunk_idx : int
        Global index of the key/value chunk.

    Returns
    -------
    int
        Seed of the dropout mask.
    """
    return (
        base_seed ^ (local_rank * 1_000_003) ^ (global_chunk_idx * 999_999_937)
    ) % _SEED_MODULUS


#: Ring q/k/v/out layouts. ``"bhsd"`` is the ring's historical convention (SDPA /
#: math cores); ``"bshd"`` is what the TE cores speak natively, so the TE ring
#: carries it end to end and skips every per-chunk transpose. Set on
#: :class:`~jigsaw.modules.RingAttentionConfig.core_layout`.
_SEQ_DIM = {"bhsd": 2, "bshd": 1}


def _seq_dim(layout: str) -> int:
    """Axis holding the sequence in a ring q/k/v/out tensor of ``layout``."""
    return _SEQ_DIM[layout]


def _bhnd(shape: torch.Size, layout: str) -> tuple[int, int, int, int]:
    """``(b, h, n, d)`` from a ring q/k/v/out ``shape`` in ``layout``."""
    b, x, y, d = shape
    return (b, y, x, d) if layout == "bshd" else (b, x, y, d)


def _compute_kv(
    xj: torch.Tensor,
    w_qkv: torch.Tensor,
    b: int,
    n_local: int,
    h: int,
    d: int,
    layout: str = "bhsd",
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
    layout : str
        ``"bhsd"`` (default) or ``"bshd"`` -- the layout of the returned
        ``kj`` / ``vj`` (and the stacked ``kv``, minus its leading pair axis).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The fused key-value tensor and its key and value views.
    """
    kv2 = (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d)
    # (b, n, 2, h, d) -> (2, b, h, n, d)  [bhsd]  /  (2, b, n, h, d)  [bshd]
    perm = (2, 0, 1, 3, 4) if layout == "bshd" else (2, 0, 3, 1, 4)
    kv = kv2.permute(*perm).contiguous()
    return kv, kv[0], kv[1]


# --------------------------------------------------------------------------- #
# hybrid (intra-node + inter-node) ring topology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HybridTopology:
    """Layout of a 1-D ring split into nodes of equal size.

    The ring's ranks (``group.size()`` of them) are cut into ``n_nodes``
    contiguous nodes of ``gpus_per_node`` ranks each. Ring position ``p``
    (``0 <= p < group.size()``) owns global sequence chunk ``p`` and lives at
    ``(p // gpus_per_node, p % gpus_per_node)``. Every peer lookup a hybrid ring
    needs -- the next rank along the intra-node ring, the next along the
    inter-node ring, or the rank at an arbitrary ``(node, local)`` -- is a table
    lookup here, so callers never open-code ``// gpus_per_node`` /
    ``% gpus_per_node`` arithmetic.

    ``intranode_group`` / ``internode_group`` are the sub-groups containing this
    rank, and are what every one-level hop -- the intra-node ring/broadcast and
    the inter-node ring alike -- actually communicates over, so the two levels
    sit on separate NCCL communicators and can progress independently. Each is
    built in the obvious order (:func:`get_hybrid_topology`), so this rank's
    position is ``local_idx`` in ``intranode_group`` and ``node_idx`` in
    ``internode_group``; :meth:`intranode_peer` / :meth:`internode_peer` return
    peers in that space.

    :meth:`intra_peer` / :meth:`inter_peer` / :meth:`rank_at` stay in ring-position
    space over the *full* group -- still needed by hops that move along both axes
    at once (the backward dK/dV relay, which hands a partial straight to its
    owner across nodes *and* local indices) and by global-chunk-id arithmetic.
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
        """Return the position of this rank in the ring."""
        return self.node_idx * self.gpus_per_node + self.local_idx

    def rank_at(self, node: int, local: int) -> int:
        """Return the ring position at ``(node, local)``, both taken modulo their extent."""
        return self._grid[node % self.n_nodes][local % self.gpus_per_node]

    def intra_peer(self, offset: int) -> int:
        """Return the ring position ``offset`` steps along the intra-node ring."""
        return self.rank_at(self.node_idx, self.local_idx + offset)

    def inter_peer(self, offset: int) -> int:
        """Return the ring position ``offset`` steps along the inter-node ring."""
        return self.rank_at(self.node_idx + offset, self.local_idx)

    def intranode_peer(self, offset: int) -> int:
        """Rank *within* ``intranode_group``, ``offset`` steps along the node's ring.

        ``intranode_group`` lists this node's ranks in node-local-index order, so
        its rank ``i`` is node-local index ``i`` and this rank sits at
        ``local_idx``.
        """
        return (self.local_idx + offset) % self.gpus_per_node

    def internode_peer(self, offset: int) -> int:
        """Rank *within* ``internode_group``, ``offset`` steps along the node ring.

        ``internode_group`` lists its members in node order, so its rank ``i`` is
        node ``i`` and this rank sits at ``node_idx``.
        """
        return (self.node_idx + offset) % self.n_nodes


_HYBRID_TOPOLOGY_CACHE: dict[tuple[dist.ProcessGroup, int], HybridTopology] = {}


def get_hybrid_topology(group: dist.ProcessGroup, gpus_per_node: int) -> HybridTopology:
    """Build and cache the :class:`HybridTopology` of a ring.

    Parameters
    ----------
    group : dist.ProcessGroup
        Process group of the ring.
    gpus_per_node : int
        Number of consecutive ring ranks that share a node.

    Returns
    -------
    HybridTopology
        Topology of the ring.

    Raises
    ------
    ValueError
        If ``gpus_per_node`` is not positive or does not divide the ring size.
    """
    key = (group, gpus_per_node)
    cached = _HYBRID_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    if gpus_per_node <= 0:
        raise ValueError("gpus_per_node must be positive")
    # Global ranks of this ring's members, ordered by ring position.
    ring_global_ranks = dist.get_process_group_ranks(group)
    ring_group_size = len(ring_global_ranks)
    if ring_group_size % gpus_per_node:
        raise ValueError("ring size must be divisible by gpus_per_node")
    n_nodes = ring_group_size // gpus_per_node

    # `grid` is in ring-position space (0 .. ring_group_size - 1); the P2P paths
    # address peers by ring position over `group`. Ring position p lives at
    # (p // gpus_per_node, p % gpus_per_node).
    grid = tuple(
        tuple(node * gpus_per_node + local for local in range(gpus_per_node))
        for node in range(n_nodes)
    )

    node_idx, local_idx = divmod(dist.get_rank(group), gpus_per_node)

    # Build only the two sub-groups this rank belongs to. Every other member of
    # each sub-group derives the identical rank list, and
    # `use_local_synchronization=True` keeps the barrier local to those members,
    # so the rings at other mesh coordinates stay independent.
    #
    # Member order is load-bearing: `intranode_group` is listed in node-local
    # index order and `internode_group` in node order, which is what lets
    # `HybridTopology.intranode_peer` / `.internode_peer` return a sub-group rank
    # as plain modular arithmetic on `local_idx` / `node_idx`.
    intranode_group = dist.new_group(
        ranks=[ring_global_ranks[p] for p in grid[node_idx]],
        use_local_synchronization=True,
    )
    internode_group = dist.new_group(
        ranks=[ring_global_ranks[grid[node][local_idx]] for node in range(n_nodes)],
        use_local_synchronization=True,
    )
    # This rank is a member of both sub-groups, so neither is NON_GROUP_MEMBER.
    assert isinstance(intranode_group, dist.ProcessGroup)
    assert isinstance(internode_group, dist.ProcessGroup)
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
    """Attributes stored on the autograd context."""

    scale: float
    local_rank: int
    group: dist.ProcessGroup
    ring_group_size: int
    config: RingAttentionConfig
    dropout_p: float
    base_seed: Optional[int]
    n_chunks: int
    saved_tensors: tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple
    ]


# --------------------------------------------------------------------------- #
# autograd.Function glue
# --------------------------------------------------------------------------- #
class _RingAttention(torch.autograd.Function):
    """Ring attention autograd function."""

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
        local_rank: int,
        config: RingAttentionConfig,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, Optional[int], tuple]:
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
            Softmax scale.
        dropout_p : float
            Dropout probability.
        group : dist.ProcessGroup
            Process group of the ring.
        local_rank : int
            Rank within the process group.
        config : RingAttentionConfig
            Ring-attention configuration.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple]
            Output, log-sum-exp, saved chunks, dropout base seed (``None``
            without dropout) and per-chunk RNG states.
        """
        ring_group_size = dist.get_world_size(group)
        from .forward import ring_forward

        out, lse, chunks_to_save, base_seed, rng_states = ring_forward(
            q,
            k,
            v,
            x,
            w_qkv,
            scale,
            dropout_p,
            group,
            local_rank,
            ring_group_size,
            config,
        )
        return out, lse, chunks_to_save, base_seed, rng_states

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
        q, k, v, x, w_qkv, scale, dropout_p, group, local_rank, config = inputs
        out, lse, chunks_to_save, base_seed, rng_states = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)
        non_none_rng_states = [r for r in rng_states if r is not None]
        if non_none_rng_states:
            ctx.mark_non_differentiable(*non_none_rng_states)
        c = cast(RingCtx, ctx)
        c.n_chunks = len(chunks_to_save)
        ctx.save_for_backward(
            q,
            out,
            x,
            # `(b, h, s)`, exactly as `ring_forward` produced it and as every
            # inner core's backward expects it. (Was saved `.mT`'d to `(b, s, h)`
            # here; `ring_backward` passes it straight through, so the cores got
            # the transposed tensor -- a silent stride error for SDPA/TE, a hard
            # shape error for the math core, whenever heads != seq_local.)
            lse,
            w_qkv,
            *chunks_to_save,  # *[t for pair in chunks_to_save for t in pair]
            *rng_states,
        )
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.ring_group_size = dist.get_world_size(
            group
        )  # This is recomputed in forward and in setup context
        c.config = config
        c.dropout_p = dropout_p
        c.base_seed = base_seed

    @staticmethod
    def backward(
        ctx: RingCtx,
        dout: torch.Tensor,
        grad_l: torch.Tensor,
        grad_kv_chunks: list,
        grad_base_seed: None,
        grad_rng_states: list,
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
    ]:
        """Compute gradients for the ring-attention forward pass.

        Parameters
        ----------
        ctx : RingCtx
            Autograd context containing saved tensors and configuration.
        dout : torch.Tensor
            Gradient of the attention output.
        grad_l : torch.Tensor
            Gradient placeholder for the log-sum-exp.
        grad_kv_chunks : list
            Gradient placeholders for the saved chunks.
        grad_base_seed : None
            Gradient placeholder for the dropout base seed.
        grad_rng_states : list
            Gradient placeholders for the per-chunk RNG states.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Gradients for differentiable inputs followed by ``None`` for
            non-differentiable inputs.
        """
        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        n_chunks = ctx.n_chunks
        chunks_to_save = ctx.saved_tensors[5 : 5 + n_chunks]
        rng_states = ctx.saved_tensors[5 + n_chunks :]

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
            ctx.ring_group_size,
            cast(tuple[torch.Tensor, ...], chunks_to_save),
            ring_config=ctx.config,
            dropout_p=ctx.dropout_p,
            base_seed=ctx.base_seed,
            rng_states=cast(tuple[Optional[torch.Tensor], ...], rng_states),
        )

        return dq, dk, dv, None, None, None, None, None, None, None
