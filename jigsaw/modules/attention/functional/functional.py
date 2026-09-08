"""Ring attention forward-backward implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, cast

import torch
import torch.distributed as dist

from ..config import RingAttentionConfig

_SEED_MODULUS = 2**31 - 1


def combine_dropout_seed(base_seed: int, local_rank: int, global_chunk_idx: int) -> int:
    """Deterministic per-(rank, global chunk) sub-seed derived from one shared ``base_seed``.

    Lets the ring's per-chunk dropout core (:mod:`.cores.math`) regenerate the
    identical mask in the forward and backward loops from three plain integers
    instead of saving a mask/RNG-state tensor per ring step -- mirroring how
    the SDPA/TE cores instead thread back the *actual* opaque RNG state their
    kernels generate (see :func:`.forward.ring_forward` /
    :func:`.backward.ring_backward`). ``base_seed`` is drawn once per
    ring-attention call and identifies the pass; ``global_chunk_idx`` -- not
    the raw ring step -- identifies the (query-owning-rank, key/value chunk)
    pairing, since forward and backward may visit the same pairing at
    different loop steps.
    """
    return (
        base_seed ^ (local_rank * 1_000_003) ^ (global_chunk_idx * 999_999_937)
    ) % _SEED_MODULUS


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

    The ring's ranks (``group.size()`` of them) are cut into ``n_nodes``
    contiguous nodes of ``gpus_per_node`` ranks each. Ring position ``p``
    (``0 <= p < group.size()``) owns global sequence chunk ``p`` and lives at
    ``(p // gpus_per_node, p % gpus_per_node)``. Every peer lookup a hybrid ring
    needs -- the next rank along the intra-node ring, the next along the
    inter-node ring, or the rank at an arbitrary ``(node, local)`` -- is a table
    lookup here, so callers never open-code ``// gpus_per_node`` /
    ``% gpus_per_node`` arithmetic.

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

    ``group`` is the ring itself -- typically one dimension of a larger device
    mesh (e.g. the ``row`` group of a ``(dp, row, col)`` mesh), so the ring's
    world size is ``group.size()``, *not* the global world size. Ring position
    ``p`` (``0 <= p < group.size()``) is the rank ``p`` of ``group``; it maps to
    global rank ``dist.get_process_group_ranks(group)[p]``.

    The intra-/inter-node sub-groups are created with
    ``use_local_synchronization=True`` so that the rings living at the other mesh
    coordinates can build their own sub-groups concurrently -- only the members
    of each sub-group synchronise, so ranks outside this ring never have to enter
    (or even know about) these ``new_group`` calls. The result is cached per
    ``(group, gpus_per_node)``; the group object is the key directly so a freed
    process group can never alias a live one (as ``id(group)`` could).

    Parameters
    ----------
    group : dist.ProcessGroup
        The ring's process group (one mesh dimension, or the whole world).
    gpus_per_node : int
        Number of consecutive ring ranks that share a node.

    Returns
    -------
    HybridTopology

    Raises
    ------
    ValueError
        If ``gpus_per_node`` is not positive or the ring size
        (``group.size()``) is not divisible by ``gpus_per_node``.
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
    intranode_group = dist.new_group(
        ranks=[ring_global_ranks[p] for p in grid[node_idx]],
        use_local_synchronization=True,
    )
    internode_group = dist.new_group(
        ranks=[ring_global_ranks[grid[node][local_idx]] for node in range(n_nodes)],
        use_local_synchronization=True,
    )
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
            Attention scaling factor.
        dropout_p : float
            Attention dropout probability.
        group : dist.ProcessGroup
            Distributed process group used for ring communication.
        local_rank : int
            Rank within the process group.
        config : RingAttentionConfig
            Ring-attention configuration.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, tuple]
            Attention output, log-sum-exp values, saved chunks, the dropout
            base seed (``None`` if ``dropout_p == 0``), and per-global-chunk
            RNG state for cores that need it replayed verbatim (``.sdpa`` /
            ``.te``; all ``None`` for ``.math``, which rederives its mask from
            the base seed instead -- see :mod:`.cores`).
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
            lse.mT,
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
            Gradient placeholder for the non-differentiable log-sum-exp output.
        grad_kv_chunks : list
            Gradient placeholders for saved non-differentiable chunks.
        grad_base_seed : None
            Gradient placeholder for the non-differentiable (and non-tensor)
            dropout base seed.
        grad_rng_states : list
            Gradient placeholders for the saved (non-differentiable) per-chunk
            RNG state.

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
