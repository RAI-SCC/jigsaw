"""Collectives that stay correct when called from a functorch-vmapped backward.

``XWT.backward`` (and ``fused_xw_xtw_twoway``) run *inside* ``vmap`` when a
Bayesian ensemble is differentiated with ``vmap(grad(...))``. The ``c10d``
collectives they need have two failure modes under a transform:

* the legacy ``dist.*`` ops (``all_to_all`` / ``all_reduce``) register no
  ``functorch`` batching rule and raise outright;
* the lowered ``_c10d_functional`` ops *do* fall back to a generic per-slice
  loop -- but that fallback issues the collective **asynchronously** for every
  ensemble slice and the trailing ``wait_tensor`` on the last slice does not
  actually block, so the last ensemble member reads a half-finished buffer
  (observed: only the final member's ``qkv.weight`` / bias grad drifts, by
  whole units).

The fix is the same fold-into-batch trick the ``te`` / ring cores use: a custom
``autograd.Function`` whose ``vmap`` staticmethod *absorbs* the vmapped axis --
inside it the tensors are plain (the ensemble axis is just a leading dim), so a
single synchronous collective runs with no per-slice fallback and its wait
genuinely blocks. Outside a transform these are a thin pass-through to one
blocking functional collective.
"""

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as _funcol


def _blocking_all_reduce_sum(t: torch.Tensor, group_name: str) -> torch.Tensor:
    out = torch.ops._c10d_functional.all_reduce(t.contiguous(), "sum", group_name)
    return torch.ops._c10d_functional.wait_tensor(out)


def _blocking_all_gather(t: torch.Tensor, world_size: int, group_name: str) -> torch.Tensor:
    out = torch.ops._c10d_functional.all_gather_into_tensor(
        t.contiguous(), world_size, group_name
    )
    return torch.ops._c10d_functional.wait_tensor(out)


class _AllReduceSum(torch.autograd.Function):
    """SUM all-reduce ``t`` over ``group`` that survives an enclosing ``vmap``."""

    @staticmethod
    def forward(t: torch.Tensor, group_name: str) -> torch.Tensor:  # type: ignore[override]
        return _blocking_all_reduce_sum(t.detach(), group_name)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Optional[torch.Tensor], None]:
        # Only ever used to reduce an already-computed gradient in a backward
        # pass; the result is not itself differentiated.
        return None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple,
        t: torch.Tensor,
        group_name: str,
    ) -> tuple[torch.Tensor, int]:
        t_bdim = in_dims[0]
        if t_bdim is None:
            t = t.expand(info.batch_size, *t.shape)
        else:
            t = t.movedim(t_bdim, 0)
        # ``t`` is now a plain tensor with the ensemble axis at 0. all-reduce is
        # elementwise, so one synchronous call over the whole tensor is exactly
        # a per-member all-reduce -- and the wait here really blocks.
        return _blocking_all_reduce_sum(t.detach(), group_name), 0


class _PeerExchange(torch.autograd.Function):
    """Return the single channel-parallel peer's copy of ``x`` (2-way split).

    Replaces the ``all_to_all`` swap in :func:`fused_xw_xtw_twoway` on the
    transform path.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        x: torch.Tensor, mod: int, group_name: str
    ) -> torch.Tensor:
        g = _blocking_all_gather(x.detach().unsqueeze(0), 2, group_name)
        return g[1 - mod]

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        pass

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[None, None, None]:
        return None, None, None

    @staticmethod
    def vmap(
        info: Any,
        in_dims: tuple,
        x: torch.Tensor,
        mod: int,
        group_name: str,
    ) -> tuple[torch.Tensor, int]:
        x_bdim = in_dims[0]
        if x_bdim is None:
            x = x.expand(info.batch_size, *x.shape)
        else:
            x = x.movedim(x_bdim, 0)
        # x: [members, ...]. Gather a leading size-1 axis so the result is
        # [2, members, ...] (rank-major); pick the peer's block.
        g = _blocking_all_gather(x.detach().unsqueeze(0), 2, group_name)
        return g[1 - mod], 0


def vmap_safe_all_reduce_sum(
    t: torch.Tensor, group: "dist.ProcessGroup"
) -> torch.Tensor:
    """SUM all-reduce ``t`` over ``group`` (see module docstring)."""
    group_name = _funcol._resolve_group_name(group)
    return _AllReduceSum.apply(t, group_name)


def vmap_safe_peer_exchange(
    x: torch.Tensor, mod: int, group: "dist.ProcessGroup"
) -> torch.Tensor:
    """Return the 2-way channel peer's copy of ``x`` (see module docstring)."""
    group_name = _funcol._resolve_group_name(group)
    return _PeerExchange.apply(x.contiguous(), mod, group_name)
