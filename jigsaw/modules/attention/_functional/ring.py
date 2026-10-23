"""Ring attention autograd function."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

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


#: Sequence axis per ring layout: ``"bhsd"`` (SDPA / math) or ``"bshd"`` (TE).
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
        ``"bhsd"`` (default) or ``"bshd"``: the layout of the returned
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
# hybrid (intra-group + inter-group) ring topology
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HybridTopology:
    """Layout of a 1-D ring split into equal intra groups.

    Each ``intra_group`` has ``intra_size`` consecutive ring ranks, and
    ``inter_group`` joins the ranks at the same local index across the
    ``n_groups`` intra groups. Ring position ``p`` owns sequence chunk ``p`` and
    sits at ``(p // intra_size, p % intra_size)``.

    The peer methods take ``offset``, the number of steps along a ring (negative
    goes backwards). :meth:`intra_peer`, :meth:`inter_peer` and :meth:`rank_at`
    return ring positions over the full group; :meth:`intra_group_peer` and
    :meth:`inter_group_peer` return ranks within the sub-groups.
    """

    n_groups: int
    intra_size: int
    group_idx: int
    local_idx: int
    intra_group: dist.ProcessGroup
    inter_group: dist.ProcessGroup
    _grid: tuple[tuple[int, ...], ...]

    @property
    def ring_rank(self) -> int:
        """Return the position of this rank in the ring.

        Returns
        -------
        int
            Ring position, ``group_idx * intra_size + local_idx``.
        """
        return self.group_idx * self.intra_size + self.local_idx

    def rank_at(self, group_idx: int, local: int) -> int:
        """Return the ring position at ``(group_idx, local)``, both taken modulo their extent.

        Parameters
        ----------
        group_idx : int
            Intra-group index along the outer ring.
        local : int
            Index within the intra group.

        Returns
        -------
        int
            Ring position over the full group.
        """
        return self._grid[group_idx % self.n_groups][local % self.intra_size]

    def intra_peer(self, offset: int) -> int:  # numpydoc ignore=PR01,RT01
        """Return the ring position ``offset`` steps along the intra-group ring."""
        return self.rank_at(self.group_idx, self.local_idx + offset)

    def inter_peer(self, offset: int) -> int:  # numpydoc ignore=PR01,RT01
        """Return the ring position ``offset`` steps along the outer ring."""
        return self.rank_at(self.group_idx + offset, self.local_idx)

    def intra_group_peer(self, offset: int) -> int:  # numpydoc ignore=PR01,RT01
        """Return the rank in ``intra_group`` ``offset`` steps along its ring."""
        return (self.local_idx + offset) % self.intra_size

    def inter_group_peer(self, offset: int) -> int:  # numpydoc ignore=PR01,RT01
        """Return the rank in ``inter_group`` ``offset`` steps along its ring."""
        return (self.group_idx + offset) % self.n_groups


_HYBRID_TOPOLOGY_CACHE: dict[tuple[dist.ProcessGroup, int], HybridTopology] = {}


def get_hybrid_topology(group: dist.ProcessGroup, intra_size: int) -> HybridTopology:
    """Build and cache the :class:`HybridTopology` of a ring.

    Parameters
    ----------
    group : dist.ProcessGroup
        Process group of the ring.
    intra_size : int
        Size of each intra group.

    Returns
    -------
    HybridTopology
        Topology of the ring.

    Raises
    ------
    ValueError
        If ``intra_size`` is not positive or does not divide the ring size.
    """
    key = (group, intra_size)
    cached = _HYBRID_TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached

    if intra_size <= 0:
        raise ValueError("intra_size must be positive")
    # Global ranks of this ring's members, ordered by ring position.
    ring_global_ranks = dist.get_process_group_ranks(group)
    ring_group_size = len(ring_global_ranks)
    if ring_group_size % intra_size:
        raise ValueError("ring size must be divisible by intra_size")
    n_groups = ring_group_size // intra_size

    # `grid` is in ring-position space (0 .. ring_group_size - 1); the P2P paths
    # address peers by ring position over `group`. Ring position p lives at
    # (p // intra_size, p % intra_size).
    grid = tuple(
        tuple(g * intra_size + local for local in range(intra_size))
        for g in range(n_groups)
    )

    group_idx, local_idx = divmod(dist.get_rank(group), intra_size)

    # Build only the two sub-groups this rank belongs to; local synchronization
    # keeps rings at other mesh coordinates independent.
    #
    # Member order matters: rank i of `intra_group` is local index i, and
    # rank i of `inter_group` is group i. That is why `intra_group_peer` /
    # `inter_group_peer` can compute a peer's sub-group rank as
    # `(local_idx + offset) % intra_size` / `(group_idx + offset) % n_groups`.
    intra_group = dist.new_group(
        ranks=[ring_global_ranks[p] for p in grid[group_idx]],
        use_local_synchronization=True,
    )
    inter_group = dist.new_group(
        ranks=[ring_global_ranks[grid[g][local_idx]] for g in range(n_groups)],
        use_local_synchronization=True,
    )
    # This rank is a member of both sub-groups, so neither is NON_GROUP_MEMBER.
    assert isinstance(intra_group, dist.ProcessGroup)
    assert isinstance(inter_group, dist.ProcessGroup)
    topo = HybridTopology(
        n_groups=n_groups,
        intra_size=intra_size,
        group_idx=group_idx,
        local_idx=local_idx,
        intra_group=intra_group,
        inter_group=inter_group,
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
    base_seed: int | None
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
    ) -> tuple[torch.Tensor, torch.Tensor, tuple, int | None, tuple]:
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
        tuple[torch.Tensor, torch.Tensor, tuple, int | None, tuple]
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
            # `(b, h, s)` as the inner cores' backward expects; don't transpose.
            lse,
            w_qkv,
            *chunks_to_save,
            *rng_states,
        )
        c.scale = scale
        c.local_rank = local_rank
        c.group = group
        c.ring_group_size = dist.get_world_size(group)
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
            rng_states=cast(tuple[torch.Tensor | None, ...], rng_states),
        )

        return dq, dk, dv, None, None, None, None, None, None, None
