"""Config for attention."""

from enum import Enum
from typing import Any, Callable, Optional, cast

import torch
import torch.distributed as dist

import jigsaw.comm as comm
from jigsaw.offload import NVLinkActivationOffloader


# --------------------------------------------------------------------------- #
# enums
# --------------------------------------------------------------------------- #
class SendForward(Enum):
    """Enum for sending X or KV."""

    X = 0
    KV = 1


class SaveBackward(Enum):
    """Enum for saving X or KV."""

    X = 0
    KV = 1


class SaveScope(Enum):
    """Enum for saving X or KV locally or globally."""

    LOCAL = 0
    GLOBAL = 1


class Offload(Enum):
    """Enum for offloading to CPU."""

    DISABLED = 0
    CPU = 1


class SendBackward(Enum):
    """Enum for accumulating DKV in the backward pass."""

    SEND_DKV = 0
    GATHER_DKV = 1


class SendOp(Enum):
    """Enum for communication operation."""

    ALLTOALL = 0
    P2P = 1
    BROADCAST = 2
    HYBRID_P2P = 3
    HYBRID_BROADCAST = 4


# --------------------------------------------------------------------------- #
# attention cores
# --------------------------------------------------------------------------- #
def _attn_core_flash(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, dropout_p: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash attention forward pass. Returns (out, logsumexp)."""
    q, k, v = [x.to(torch.float16) for x in (q, k, v)]
    out, logsumexp, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        q,
        k,
        v,
        dropout_p=dropout_p,
        is_causal=False,
        scale=scale,
        return_debug_mask=False,
    )

    return out, logsumexp


def _attn_core_math(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: Optional[float] = 0.0,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vanilla attention forward pass. Returns (out, logsumexp)."""
    scale = scale or (q.size(-1) ** -0.5)
    scores = (q @ k.mT) * scale
    logsumexp = scores.logsumexp(dim=-1)  # (B, H, Sq)
    out = torch.softmax(scores, dim=-1) @ v
    return out, logsumexp


def _attn_core_flash_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    max_q: float,
    max_k: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward call to flash attention. Returns (dq, dk, dv)."""
    dq, dk, dv = torch.ops.aten._scaled_dot_product_flash_attention_backward(
        grad_out,
        q,
        k,
        v,
        out,
        logsumexp,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
        cum_seq_q=None,
        cum_seq_k=None,
        max_q=max_q,
        max_k=max_k,
        philox_seed=torch.zeros(1, dtype=torch.long, device=q.device),
        philox_offset=torch.zeros(1, dtype=torch.long, device=q.device),
    )
    return dq, dk, dv


def _attn_core_math_backward(
    grad_out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    scale: float,
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla attention backward pass using saved (global) LSE. Returns (dq, dk, dv)."""
    attn = (q @ k.mT) * scale  # (B, H, Sq, Sk)
    p = torch.exp(attn - logsumexp.unsqueeze(-1))  # global norm via saved lse

    dv = p.mT @ grad_out

    dp = grad_out @ v.mT
    d = (grad_out * out).sum(dim=-1, keepdim=True)  # GLOBAL row-sum, from saved out
    d_attn = p * (dp - d) * scale

    dq = d_attn @ k
    dk = d_attn.mT @ q
    return dq, dk, dv


# --------------------------------------------------------------------------- #
# forward comm fn selection
# --------------------------------------------------------------------------- #
def _select_forward_comm_fns(
    send_op: SendOp, send_in_forward: SendForward, send_op_fn: Callable[..., Any]
) -> tuple[Callable, Callable]:
    """
    Define send and receive functions in the forward pass.

    Parameters
    ----------
    send_op: SendOp
        Enumeration that defines what type of collective call
        is made in the forward pass.
    send_in_forward: SendForward
        Enumeration to define if X or KV are communicated in the forward.
    send_op_fn: Callable
        Function handle to the "ring" communication.

    Returns
    -------
    send_func, recv_func: tuple[Callable, Callable]
        Two function handles for sending and receiving the ring information.
    """
    send_func: Callable[..., tuple[torch.Tensor, dist.Work]]

    match send_op:
        case SendOp.P2P | SendOp.ALLTOALL:
            match send_in_forward:
                case SendForward.X:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        direction: int,
                        group: dist.ProcessGroup,
                        tag: int,
                        empty: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        direction : int
                            Direction of the ring communication.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        tag : int
                            Communication tag.
                        empty : torch.Tensor
                            Empty tensor used by communication helpers.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = torch.empty_like(x)
                        recv_buf, handle = send_op_fn(
                            x,
                            recv_buf,
                            direction,
                            group=group,
                            async_op=True,
                            tag=tag,
                            empty=empty,
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        direction: int,
                        group: dist.ProcessGroup,
                        tag: int,
                        empty: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        direction : int
                            Direction of the ring communication.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        tag : int
                            Communication tag.
                        empty : torch.Tensor
                            Empty tensor used by communication helpers.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = torch.empty_like(kv)
                        recv_buf, handle = send_op_fn(
                            kv,
                            recv_buf,
                            direction,
                            group=group,
                            async_op=True,
                            empty=empty,
                            tag=tag,
                        )
                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {send_in_forward} not implemented."
                    )

        case SendOp.BROADCAST:
            match send_in_forward:
                case SendForward.X:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        src_rank: int,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        local_rank : int
                            Rank of the current process.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        src_rank : int
                            Source rank for the broadcast.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = x if src_rank == local_rank else torch.empty_like(x)
                        handle = dist.broadcast(
                            recv_buf, src=src_rank, group=group, async_op=True
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        local_rank: int,
                        group: dist.ProcessGroup,
                        src_rank: int,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        local_rank : int
                            Rank of the current process.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        src_rank : int
                            Source rank for the broadcast.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = (
                            kv if src_rank == local_rank else torch.empty_like(kv)
                        )
                        handle = dist.broadcast(
                            recv_buf, src=src_rank, group=group, async_op=True
                        )
                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {send_in_forward} not implemented."
                    )
        case SendOp.HYBRID_P2P:
            match send_in_forward:
                case SendForward.X:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        direction: int,
                        group: dist.ProcessGroup,
                        tag: int,
                        empty: torch.Tensor,
                        group_size: int = 4,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        direction : int
                            Direction of the ring communication.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        tag : int
                            Communication tag.
                        empty : torch.Tensor
                            Empty tensor used by communication helpers.
                        group_size : int
                            Number of ranks in the communication group.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = torch.empty_like(x)
                        recv_buf, handle = send_op_fn(
                            x,
                            recv_buf,
                            direction,
                            group=group,
                            async_op=True,
                            tag=tag,
                            empty=empty,
                            group_size=group_size,
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(  # type: ignore[assignment]
                        x: torch.Tensor,
                        kv: torch.Tensor,
                        direction: int,
                        group: dist.ProcessGroup,
                        tag: int,
                        empty: torch.Tensor,
                        *args: Any,
                        **kwargs: Any,
                    ) -> tuple[torch.Tensor, dist.Work]:
                        """Send the configured tensor to the next ring participant.

                        Parameters
                        ----------
                        x : torch.Tensor
                            Input tensor.
                        kv : torch.Tensor
                            Fused key-value tensor.
                        direction : int
                            Direction of the ring communication.
                        group : dist.ProcessGroup
                            Process group used for distributed communication.
                        tag : int
                            Communication tag.
                        empty : torch.Tensor
                            Empty tensor used by communication helpers.

                        args : tuple[Any, ...]
                            Additional positional arguments.
                        kwargs : dict[str, Any]
                            Additional keyword arguments.

                        Returns
                        -------
                        tuple[torch.Tensor, dist.Work]
                            Result returned by the operation.
                        """
                        recv_buf = torch.empty_like(kv)
                        recv_buf, handle = send_op_fn(
                            kv,
                            recv_buf,
                            direction,
                            group=group,
                            async_op=True,
                            empty=empty,
                            tag=tag,
                        )
                        return recv_buf, handle
        case _:
            raise NotImplementedError(f"send_op option {send_op} not implemented.")

    match send_in_forward:
        case SendForward.X:

            def recv_func(
                recv_buf: torch.Tensor,
                xj: torch.Tensor,
                kv_j: torch.Tensor,
                kj: torch.Tensor,
                vj: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                """Update the received ring tensors.

                Parameters
                ----------
                recv_buf : torch.Tensor
                    Tensor buffer receiving communicated data.
                xj : torch.Tensor
                    Input sequence chunk.
                kv_j : torch.Tensor
                    Fused key-value tensor for the current ring chunk.
                kj : torch.Tensor
                    Key tensor for the current ring chunk.
                vj : torch.Tensor
                    Value tensor for the current ring chunk.

                Returns
                -------
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                    Result returned by the operation.
                """
                return recv_buf, kv_j, kj, vj

        case SendForward.KV:

            def recv_func(
                recv_buf: torch.Tensor,
                xj: torch.Tensor,
                kv_j: torch.Tensor,
                kj: torch.Tensor,
                vj: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                """Update the received ring tensors.

                Parameters
                ----------
                recv_buf : torch.Tensor
                    Tensor buffer receiving communicated data.
                xj : torch.Tensor
                    Input sequence chunk.
                kv_j : torch.Tensor
                    Fused key-value tensor for the current ring chunk.
                kj : torch.Tensor
                    Key tensor for the current ring chunk.
                vj : torch.Tensor
                    Value tensor for the current ring chunk.

                Returns
                -------
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
                    Result returned by the operation.
                """
                kv_j = recv_buf
                return xj, kv_j, kv_j[0], kv_j[1]

        case _:
            raise NotImplementedError(
                f"send_in_forward option {send_in_forward} not implemented."
            )

    return send_func, recv_func


def _select_save_for_backward_fn(
    save_for_backward: SaveBackward,
    save_scope: SaveScope,
    offloader: Optional[NVLinkActivationOffloader],
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the function for saving tensors for backward based on the config."""
    match save_for_backward:
        case SaveBackward.X:
            if isinstance(offloader, NVLinkActivationOffloader):
                return lambda x, kv: offloader.offload(x)
            # keep on GPU -- must clone if Xj will be overwritten by recv next iter
            if save_scope == SaveScope.GLOBAL:
                return lambda x, kv: x.clone()
            return lambda x, kv: x

        case SaveBackward.KV:
            # KV is already contiguous, shape [2, B, H, N, d]
            if isinstance(offloader, NVLinkActivationOffloader):

                def offload_kv(x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
                    """Offload the key-value tensor for backward computation.

                    Parameters
                    ----------
                    x : torch.Tensor
                        Input tensor.
                    kv : torch.Tensor
                        Fused key-value tensor.

                    Returns
                    -------
                    torch.Tensor
                        Result returned by the operation.
                    """
                    kv_cpu = cast(torch.Tensor, offloader.offload(kv))
                    del kv
                    return kv_cpu

                return offload_kv

            if save_scope == SaveScope.GLOBAL:
                return lambda x, kv: kv.clone()
            return lambda x, kv: kv

        case _:
            return lambda xj, kv_j: None  # default no-op


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
class RingAttentionConfig:
    """Config class for ring attention.

    Bundles every policy decision (what to send, what to save, how to
    accumulate dK/dV, whether to offload) together with the concrete
    callables that implement each policy, so the forward/backward loops
    only ever call `self.config.<thing>` and never branch on raw enums
    themselves.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        send_in_forward: SendForward = SendForward.X,
        send_op: SendOp = SendOp.HYBRID_P2P,
        save_for_backward: SaveBackward = SaveBackward.KV,
        save_scope: SaveScope = SaveScope.GLOBAL,
        send_backward: SendBackward = SendBackward.SEND_DKV,
        offload: Offload = Offload.DISABLED,
        gpus_per_node: int = 4,
    ):
        """Initialize the object.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.
        send_in_forward : SendForward
            Tensor representation communicated during the forward pass.
        send_op : SendOp
            Communication operation used by the ring.
        save_for_backward : SaveBackward
            Tensor representation saved for the backward pass.
        save_scope : SaveScope
            Scope in which intermediate tensors are saved.
        send_backward : SendBackward
            Strategy used to communicate backward gradients.
        offload : Offload
            Activation offloading policy.
        gpus_per_node : int
            Number of ranks assigned to each node for hybrid communication.

        """
        if gpus_per_node <= 0:
            raise ValueError("gpus_per_node must be positive")
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.send_op = send_op
        self.save_scope = save_scope
        self.send_backward = send_backward
        self.offload = offload
        self.gpus_per_node = gpus_per_node
        self.device = device
        self.gpus_per_node = gpus_per_node

        self.offloader = self._init_offloader(offload, device)
        self.p2p_fn = self._p2p_fn(send_op)
        self.send_op_fn = self._send_op_fn(send_op, device)
        self.attn_core_forward_fn = self._select_attn_core_forward_fn(device)
        self.attn_core_backward_fn = self._select_attn_core_backward_fn(device)
        self.forward_send_fn, self.forward_recv_fn = _select_forward_comm_fns(
            send_op, send_in_forward, self.send_op_fn
        )
        self.save_for_backward_fn = _select_save_for_backward_fn(
            save_for_backward, save_scope, self.offloader
        )

    # -- selection helpers ------------------------------------------------- #
    def _get_data_to_send(self, x: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """Select the tensor representation used for communication.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        kv : torch.Tensor
            Fused key-value tensor.

        Returns
        -------
        torch.Tensor
            Result returned by the operation.
        """
        if self.send_in_forward == SendForward.X:
            return x
        elif self.send_in_forward == SendForward.KV:
            return kv

    def _init_offloader(
        self, offload: Offload, device: torch.device
    ) -> Optional[NVLinkActivationOffloader]:
        """Initialize the configured activation offloader.

        Parameters
        ----------
        offload : Offload
            Activation offloading policy.
        device : torch.device
            Device on which tensors and communication are placed.

        Returns
        -------
        Optional[NVLinkActivationOffloader]
            Result returned by the operation.
        """
        if offload == Offload.DISABLED:
            return None
        elif offload == Offload.CPU:
            return NVLinkActivationOffloader(device)
        raise ValueError(f"Unhandled offload option {offload}")

    def _p2p_fn(self, send_op: SendOp) -> Callable[..., Any]:
        """Select the point-to-point communication function.

        Parameters
        ----------
        send_op : SendOp
            Communication operation used by the ring.

        Returns
        -------
        Callable[..., Any]
            Result returned by the operation.
        """
        if dist.get_backend() == "gloo":
            return comm.comm_1d_p2p
        else:
            match send_op:
                case SendOp.ALLTOALL:
                    return comm.comm_1d_alltoall
                case SendOp.P2P:
                    return comm.comm_1d_p2p
                case SendOp.BROADCAST:
                    return comm.comm_1d_alltoall

        # Default comm function type is p2p
        return comm.comm_1d_p2p

    def _send_op_fn(self, send_op: SendOp, device: torch.device) -> Callable[..., Any]:
        """Select the configured communication function.

        Parameters
        ----------
        send_op : SendOp
            Communication operation used by the ring.
        device : torch.device
            Device on which tensors and communication are placed.

        Returns
        -------
        Callable[..., Any]
            Result returned by the operation.
        """
        if dist.get_backend() == "gloo":
            return comm.comm_1d_p2p

        match send_op:
            case SendOp.ALLTOALL | SendOp.HYBRID_P2P:
                if dist.get_backend() != "gloo":
                    return comm.comm_1d_alltoall
            case SendOp.P2P | SendOp.HYBRID_P2P:
                return comm.comm_1d_p2p
            case SendOp.BROADCAST:
                return comm.comm_broadcast

        return comm.comm_1d_p2p

    def _select_attn_core_forward_fn(self, device: torch.device) -> Callable:
        """Select the attention forward kernel for the device.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.

        Returns
        -------
        Callable
            Result returned by the operation.
        """
        if device.type == "cuda":
            return _attn_core_flash
        elif device.type == "cpu":
            return _attn_core_math
        raise ValueError(f"No valid attention core forward for device {device.type}")

    def _select_attn_core_backward_fn(self, device: torch.device) -> Callable:
        """Select the attention backward kernel for the device.

        Parameters
        ----------
        device : torch.device
            Device on which tensors and communication are placed.

        Returns
        -------
        Callable
            Result returned by the operation.
        """
        if device.type == "cuda":
            return _attn_core_flash_backward
        elif device.type == "cpu":
            return _attn_core_math_backward
        raise ValueError(f"No valid attention core backward for device {device.type}")

    # -- small shared behaviours used by both forward implementations ------ #
    def wait_offloads(self) -> None:
        """Block until any in-flight activation offloads have completed."""
        if self.offloader is not None:
            self.offloader.wait_offloads()

    def __repr__(self) -> str:
        """Return string representation of config class."""
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_op={self.send_op.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name})"
        )
