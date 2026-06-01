from collections.abc import Callable
from enum import Enum

import torch
import torch.distributed as dist

from .offload import NVLinkActivationOffloader


def _online_softmax_merge(o1, l1, o2, l2):
    l_new = torch.logaddexp(l1, l2)
    e1 = torch.exp(l1 - l_new)
    e2 = torch.exp(l2 - l_new)
    o_new = e1.unsqueeze(-1) * o1 + e2.unsqueeze(-1) * o2
    return o_new, l_new


def _ring_send_recv(
    tensor, world_size, rank, group, empty, recv_buf, forward_or_backward=0, offset=1
):
    if forward_or_backward == 0:
        send_to = (rank + offset) % world_size
        recv_from = (rank - offset) % world_size
    else:
        send_to = (rank - offset) % world_size
        recv_from = (rank + offset) % world_size
    output_list = [recv_buf if i == recv_from else empty for i in range(world_size)]
    input_list = [tensor if i == send_to else empty for i in range(world_size)]
    handle = dist.all_to_all(output_list, input_list, group=group, async_op=True)
    return recv_buf, handle


def _compute_kv(xj, w_qkv, b, n_local, h, d):
    """Compute (K, V) from X. Returns (KV, Kj, Vj). KV is the underlying buffer."""
    kv = (
        (xj @ w_qkv[d * h :, :].mT).reshape(b, n_local, 2, h, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    kj = kv[0]
    vj = kv[1]
    return kv, kj, vj


# Module-level enum definitions
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
    """Enum for backward communication."""

    SEND_DKV = 0
    GATHER_DKV = 1


class SendForwardOp(Enum):
    """Enum for forward communication operation."""

    SENDRECV = 0
    BROADCAST = 1


class RingAttentionConfig:
    """Config class for ring attention."""

    def __init__(
        self,
        send_in_forward: SendForward = SendForward.KV,
        send_forward_op: SendForwardOp = SendForwardOp.SENDRECV,
        save_for_backward: SaveBackward = SaveBackward.KV,
        save_scope: SaveScope = SaveScope.LOCAL,
        send_backward: SendBackward = SendBackward.SEND_DKV,
        offload: Offload = Offload.DISABLED,
        device: torch.device = None,
    ):
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.send_forward_op = send_forward_op
        self.save_scope = save_scope
        self.send_backward = send_backward
        self.offload = offload
        self.offloader = self.init_offloader(offload, device)

    def init_offloader(self, offload, device):
        """Initialize offloader based on config."""
        if offload == Offload.DISABLED:
            return None
        elif offload == Offload.CPU:
            return NVLinkActivationOffloader(device)

    def __repr__(self):
        """Print statement of RingAttentionConfig class."""
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"send_forward_op={self.send_forward_op.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"send_backward={self.send_backward.name}, "
            f"offload={self.offload.name})"
        )

class RingAttentionFunction(torch.autograd.Function):
    """Ring Attention autograd function."""

    @staticmethod
    def forward(q, k, v, x, w_qkv, scale, group, empty, rank, world_size, config):
        """Forward pass of ring attention."""
        b, h, n_local, d = q.shape

        world_size = dist.get_world_size(group)
        out = torch.zeros_like(q)
        lse = torch.full((b, n_local, h), float("-inf"), device=q.device, dtype=q.dtype)
        empty = torch.empty(0, device=x.device, dtype=x.dtype)

        out, lse, chunks_to_save = ring_forward(
            q, x, w_qkv, scale, group, empty, rank, world_size, config
        )

        out = out.to(torch.bfloat16)

        return out, lse, chunks_to_save

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Set up context.

        ctx: FunctionCtx
        inputs: tuple(x, w, b, local_rank, device, selector, spatial_group, bayesian)
        output: xwt
        """
        (q, k, v, x, w_qkv, scale, group, empty, local_rank, world_size, config) = (
            inputs
        )

        out, lse, chunks_to_save = output
        ctx.mark_non_differentiable(lse)
        ctx.mark_non_differentiable(*chunks_to_save)

        ctx.save_for_backward(
            q, out, x, lse.mT, w_qkv, *[t for pair in chunks_to_save for t in pair]
        )

        ctx.W_qkv = w_qkv
        ctx.empty = empty
        ctx.scale = scale
        ctx.local_rank = local_rank
        ctx.group = group
        ctx.world_size = world_size
        ctx.config = config

    @staticmethod
    def backward(ctx, dout, grad_l, grad_kv_chunks):
        """Backward pass of ring attention."""
        scale = ctx.scale
        group = ctx.group
        world_size = ctx.world_size
        empty = ctx.empty
        local_rank = ctx.local_rank
        config = ctx.config

        q, out, x, lse, w_qkv = ctx.saved_tensors[:5]
        chunks_to_save = ctx.saved_tensors[5:]

        dq, dk, dv = ring_backward(
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
            chunks_to_save,
            ring_config=config,
            empty=empty,
        )

        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def build_ring_forward(
    ring_config: RingAttentionConfig,
) -> Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        float,
        dist.ProcessGroup,
        int,
        int,
        int,
    ],
    tuple[torch.Tensor, torch.Tensor, list],
]:
    """Build the forward function for ring attention based on the config."""
    send_in_forward, recv_in_forward = _build_send_in_forward(ring_config)
    save_for_backward = _build_save_for_backward(ring_config)
    is_forward_x = ring_config.send_in_forward == SendForward.X
    is_save_scope_global = ring_config.save_scope == SaveScope.GLOBAL

    if ring_config.send_forward_op == SendForwardOp.BROADCAST:

        @torch.no_grad()
        def ring_forward(q, k, v, x, w_qkv, scale, group, empty, rank, world_size):
            """Forward pass for ring attention."""
            b, heads, n_local, d = q.shape
            world_size = dist.get_world_size(group)

            out = torch.zeros_like(q)
            lse = torch.full(
                (b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype
            )

            x_j = x

            # Local KV
            kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)]).clone().contiguous()
            kv_j, kj, vj = kv, k, v

            if not is_save_scope_global:
                chunks_to_save = [save_for_backward(x, kv)]
            else:
                chunks_to_save = [
                    None
                ] * world_size  # placeholder list to insert into for each step

            # Kick off first send
            # recv_buf either holds x or kv
            recv_buf, handle = send_in_forward(x, kv, 0, rank, group)
            handle.wait()
            x_j, kv_j, kj, vj = recv_in_forward(recv_buf, x_j, kv_j, kj, vj)

            for step in range(world_size):
                if step + 1 < world_size:
                    recv_buf, handle = send_in_forward(x, kv, step + 1, rank, group)

                if is_forward_x:
                    if step != rank:
                        # Recompute KV from current Xj (received in previous iter)
                        kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
                    else:
                        # For the step where we broadcast our own chunk, we can reuse the local KV without recomputing
                        kv_j, kj, vj = kv, k, v

                # ---- save for backward ----
                # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
                # save_local_or_global == 1: save every step's chunk

                if is_save_scope_global:
                    # append what we need to save for backward.
                    # this can be x, kv, or the pointer to cpu offloaded copies of these
                    chunks_to_save[(rank - step) % world_size] = save_for_backward(
                        x_j, kv_j
                    )

                # ---- compute flash attention on local Q vs current Kj, Vj ----
                # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
                kj_c = kj.contiguous()
                vj_c = vj.contiguous()

                # call torch sdpa
                out_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q,
                    kj_c,
                    vj_c,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    return_debug_mask=False,
                )
                out, lse = _online_softmax_merge(out, lse, out_j, l_j)
                del out_j, l_j, kj_c, vj_c

                # ---- ring rotate ----
                if step < world_size - 1:
                    handle.wait()

                    # read either x or kv values from the communicated buffers
                    # can put into a helper function when/if there are more complicated options involved
                    x_j, kv_j, kj, vj = recv_in_forward(recv_buf, x_j, kv_j, kj, vj)

            # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
            if ring_config.offloader is not None:
                ring_config.offloader.wait_offloads()

            out = out.to(torch.bfloat16)
            return out, lse, chunks_to_save
    else:

        @torch.no_grad()
        def ring_forward(q, k, v, x, w_qkv, scale, group, empty, rank, world_size):
            """Forward pass for ring attention."""
            b, heads, n_local, d = q.shape
            world_size = dist.get_world_size(group)

            out = torch.zeros_like(q)
            lse = torch.full(
                (b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype
            )

            x_j = x
            chunks_to_save = []

            # Local KV
            kv = torch.concat([k.unsqueeze(0), v.unsqueeze(0)])
            kv_j, kj, vj = kv, k, v

            # Kick off first send
            # recv_buf either holds x or kv
            recv_buf, handle = send_in_forward(
                x_j, kv_j, world_size, rank, group, empty
            )

            for step in range(world_size):
                if (step > 0) and is_forward_x:
                    # Recompute KV from current Xj (received in previous iter)
                    kv_j, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

                # ---- save for backward ----
                # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
                # save_local_or_global == 1: save every step's chunk
                should_save = is_save_scope_global or (step == 0)

                if should_save:
                    # append what we need to save for backward.
                    # this can be x, kv, or the pointer to cpu offloaded copies of these
                    chunks_to_save.append(save_for_backward(x_j, kv_j))

                # ---- compute flash attention on local Q vs current Kj, Vj ----
                # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
                kj_c = kj.contiguous()
                vj_c = vj.contiguous()

                # call torch sdpa
                out_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                    q,
                    kj_c,
                    vj_c,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    return_debug_mask=False,
                )
                out, lse = _online_softmax_merge(out, lse, out_j, l_j)
                del out_j, l_j, kj_c, vj_c

                # ---- ring rotate ----
                if step < world_size - 1:
                    handle.wait()

                    # read either x or kv values from the communicated buffers
                    # can put into a helper function when/if there are more complicated options involved
                    x_j, kv_j, kj, vj = recv_in_forward(recv_buf, x_j, kv_j, kj, vj)

                    # send x or k/v to the next rank
                    recv_buf, handle = send_in_forward(
                        x_j,
                        kv_j,
                        world_size,
                        rank,
                        group,
                        empty,
                    )

            # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
            if ring_config.offloader is not None:
                ring_config.offloader.wait_offloads()

            out = out.to(torch.bfloat16)
            return out, lse, chunks_to_save

    return ring_forward


@torch.no_grad()
def ring_backward(
    dout,
    q,
    o,
    x,
    lse,
    w_qkv,
    scale,
    group,
    local_rank,
    world_size,
    chunks_to_save,
    ring_config: RingAttentionConfig,
    empty=None,
):
    """Implement backward pass for ring attention."""
    # helper definition for implementation
    match ring_config.save_for_backward:
        case SaveBackward.X:
            x_chunks = chunks_to_save
        case SaveBackward.KV:
            kv_chunks = chunks_to_save

    b, h, seq_local, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)

    # Get buffers for dK, dV
    dk_buf, dv_buf = get_dkv_buffer(ring_config, q, world_size)

    # Pre-issue first prefetch (for save_local_or_global == 1) so H2D overlaps
    # with the compute of step 0.
    next_prefetch = None
    next_prefetch_event = None

    def _start_prefetch(idx):
        """Kick off async prefetch for chunk idx; returns (gpu_tensor, event) or None."""
        if ring_config.save_scope == SaveScope.LOCAL or ring_config.offloader is None:
            return None, None
        if idx >= len(chunks_to_save):
            return None, None
        gpu, ev = ring_config.offloader.prefetch(chunks_to_save[idx])
        return gpu, ev

    # Step 0 chunk: prefetch if applicable
    if ring_config.save_scope == SaveScope.GLOBAL and ring_config.offloader is not None:
        cur_chunk_gpu, cur_chunk_event = _start_prefetch(0)
        next_prefetch, next_prefetch_event = _start_prefetch(1)
    else:
        cur_chunk_gpu, cur_chunk_event = None, None

    # State for ring rotation
    recv_buf = None
    handle = None

    for step in range(world_size):
        # ---- obtain Kj, Vj for this step ----
        match ring_config.save_scope:
            case SaveScope.GLOBAL:
                # we have all chunks saved; use the prefetched one
                if ring_config.offloader is not None:
                    ring_config.offloader.wait_prefetch(cur_chunk_event)
                    chunk = cur_chunk_gpu
                    # kick off next prefetch ASAP to overlap with compute
                    cur_chunk_gpu, cur_chunk_event = next_prefetch, next_prefetch_event
                    next_prefetch, next_prefetch_event = _start_prefetch(step + 2)
                else:
                    chunk = chunks_to_save[step]

                if ring_config.save_for_backward == SaveBackward.KV:
                    kv = chunk
                    kj, vj = kv[0].contiguous(), kv[1].contiguous()
                elif ring_config.save_for_backward == SaveBackward.X:
                    xj = chunk
                    kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                    kj, vj = kj.contiguous(), vj.contiguous()

            case SaveScope.LOCAL:
                # save_local_or_global == 0: only own chunk saved; ring around
                if step == 0:
                    match ring_config.save_for_backward:
                        case SaveBackward.KV:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(kv_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                kv = gpu
                            else:
                                kv = kv_chunks[0]
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()
                        case SaveBackward.X:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(x_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                xj = gpu
                            else:
                                xj = x_chunks[0]
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()

                    # Kick off first ring send
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            recv_buf = torch.empty_like(x)
                            recv_buf, handle = _ring_send_recv(
                                x, world_size, local_rank, group, empty, recv_buf
                            )
                        case SendForward.KV:
                            recv_buf = torch.empty_like(kv)
                            recv_buf, handle = _ring_send_recv(
                                kv, world_size, local_rank, group, empty, recv_buf
                            )

                else:
                    # step > 0: receive next chunk
                    handle.wait()
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            xj = recv_buf
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, seq_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()
                        case SendForward.KV:
                            kv = recv_buf
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()

                    # Send either x or k,v to the next rank
                    if step < world_size - 1:
                        next_recv = torch.empty_like(recv_buf)
                        recv_buf, handle = _ring_send_recv(
                            recv_buf, world_size, local_rank, group, empty, next_recv
                        )

        # ---- flash attention backward ----
        dq_j, dk_j, dv_j = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dout,
            q,
            kj,
            vj,
            o,
            lse,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=seq_local,
            max_k=seq_local,
            philox_seed=torch.zeros(1, dtype=torch.long, device=q.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=q.device),
        )
        dq += dq_j
        del dq_j, kj, vj, kv

        # ---- accumulate dK, dV ----
        # Handles how the local calculation of dk_j, dv_j should be accumulated with the "true" value of dk, dv
        # communication is also handled here
        dk_buf, dv_buf = accumulate_dkv(
            ring_config,
            dk_buf,
            dv_buf,
            dk_j,
            dv_j,
            step,
            world_size,
            seq_local,
            local_rank,
            group,
            empty,
        )

    # return the final values for dk, dv
    dk_local, dv_local = postprocess_dkv(
        ring_config, dk_buf, dv_buf, local_rank, seq_local, group
    )

    return dq, dk_local, dv_local


def _build_send_in_forward(
    ring_config: RingAttentionConfig,
) -> tuple[
    Callable[
        [],
        tuple[torch.Tensor, dist.Work],
    ],
    Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ],
]:
    """Build the send and receive functions for forward based on the config."""
    match ring_config.send_forward_op:
        case SendForwardOp.SENDRECV:
            match ring_config.send_in_forward:
                case SendForward.X:

                    def send_func(x, kv, world_size, rank, group, empty):
                        recv_buf = torch.empty_like(x)

                        recv_buf, handle = _ring_send_recv(
                            x, world_size, rank, group, empty, recv_buf
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(x, kv, world_size, rank, group, empty):
                        recv_buf = torch.empty_like(kv)
                        recv_buf, handle = _ring_send_recv(
                            kv, world_size, rank, group, empty, recv_buf
                        )
                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {ring_config.send_in_forward} not implemented."
                    )

        case SendForwardOp.BROADCAST:
            match ring_config.send_in_forward:
                case SendForward.X:

                    def send_func(x, kv, step, rank, group):
                        if step == rank:
                            recv_buf = x
                        else:
                            recv_buf = torch.empty_like(x)
                        handle = dist.broadcast(
                            recv_buf, src=step, group=group, async_op=True
                        )
                        return recv_buf, handle

                case SendForward.KV:

                    def send_func(x, kv, step, rank, group):
                        if step == rank:
                            recv_buf = kv
                        else:
                            recv_buf = torch.empty_like(kv)
                        handle = dist.broadcast(
                            recv_buf, src=step, group=group, async_op=True
                        )
                        return recv_buf, handle

                case _:
                    raise NotImplementedError(
                        f"send_in_forward option {ring_config.send_in_forward} not implemented."
                    )
        case _:
            raise NotImplementedError(
                f"send_forward_op option {ring_config.send_forward_op} not implemented."
            )

    match ring_config.send_in_forward:
        case SendForward.X:

            def recv_func(recv_buf, xj, kv_j, kj, vj):
                return recv_buf, kv_j, kj, vj
        case SendForward.KV:

            def recv_func(recv_buf, xj, kv_j, kj, vj):
                kv_j = recv_buf
                kj = kv_j[0]
                vj = kv_j[1]
                return xj, kv_j, kj, vj
        case _:
            raise NotImplementedError(
                f"send_in_forward option {ring_config.send_in_forward} not implemented."
            )

    return send_func, recv_func


def _build_save_for_backward(
    ring_config: RingAttentionConfig,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build the function for saving tensors for backward based on the config."""
    match ring_config.save_for_backward:
        case SaveBackward.X:
            # offload X
            if ring_config.offloader is not None:
                return lambda x, kv: ring_config.offloader.offload(x)
            else:
                # keep on GPU — must clone if Xj will be overwritten by recv next iter
                if ring_config.save_scope == SaveScope.GLOBAL:
                    return lambda x, kv: x.clone()
                else:
                    return lambda x, kv: x
        case SaveBackward.KV:
            # offload KV (note: KV is already a contiguous tensor of shape [2,B,H,N,d])
            if ring_config.offloader is not None:

                def offload_kv(x, kv):
                    kv_cpu = ring_config.offloader.offload(kv)
                    del kv
                    return kv_cpu

                return offload_kv
            else:
                if ring_config.save_scope == SaveScope.GLOBAL:
                    return lambda x, kv: kv.clone()
                else:
                    return lambda x, kv: kv

        case _:
            return lambda xj, kv_j: None  # default no-op


def get_dkv_buffer(ring_config, q, world_size):
    """Get the appropriate dK, dV buffers based on the ring config."""
    # Get the dkv buffer that will be needed
    # If all-gathering the buffer, this will be where the dK, dV are accumulated
    # else, these are the buffers used to send dK and dV from rank to rank

    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:  # send dK = 1
            dk_buf = torch.zeros(
                [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]],
                device=q.device,
                dtype=q.dtype,
            )
            dv_buf = torch.zeros(
                [*q.shape[0:2], world_size * q.shape[2], *q.shape[3:]],
                device=q.device,
                dtype=q.dtype,
            )
        case SendBackward.SEND_DKV:
            dk_buf = torch.zeros(q.shape, device=q.device, dtype=q.dtype)
            dv_buf = torch.zeros(q.shape, device=q.device, dtype=q.dtype)

    return dk_buf, dv_buf


def accumulate_dkv(
    ring_config,
    dk_buf,
    dv_buf,
    dk_j,
    dv_j,
    step,
    world_size,
    seq_local,
    local_rank,
    group,
    empty,
):
    """Accumulate the locally computed dk_j, dv_j with the "true" dk, dv across the ring rotation."""
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            global_chunk_idx = (local_rank - step) % world_size
            dk_buf[
                :, :, global_chunk_idx * seq_local : (global_chunk_idx + 1) * seq_local
            ] += dk_j
            dv_buf[
                :, :, global_chunk_idx * seq_local : (global_chunk_idx + 1) * seq_local
            ] += dv_j
            del dk_j, dv_j
        case SendBackward.SEND_DKV:
            if step == 0:
                dk_buf += dk_j
                dv_buf += dv_j
                del dk_j, dv_j
            else:
                dkv = torch.cat([dk_j, dv_j], dim=0).contiguous()
                del dk_j, dv_j
                recv_buffer = torch.empty_like(dkv)
                recv_buffer, dkv_handle = _ring_send_recv(
                    dkv,
                    world_size=world_size,
                    rank=local_rank,
                    group=group,
                    empty=empty,
                    recv_buf=recv_buffer,
                    offset=step,
                    forward_or_backward=1,
                )
                dkv_handle.wait()
                del dkv
                dkj_r, dvj_r = torch.chunk(recv_buffer, 2, dim=0)
                dk_buf += dkj_r
                dv_buf += dvj_r
                del dkj_r, dvj_r, recv_buffer

    return dk_buf, dv_buf


def postprocess_dkv(ring_config, dk_buf, dv_buf, local_rank, seq_local, group):
    """Post-process the dK, dV buffers after the ring rotation is done to get the final dK, dV for the local chunk."""
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
            dk_local = dk_buf[
                :, :, local_rank * seq_local : (local_rank + 1) * seq_local
            ]
            dv_local = dv_buf[
                :, :, local_rank * seq_local : (local_rank + 1) * seq_local
            ]
        case SendBackward.SEND_DKV:
            dk_local = dk_buf
            dv_local = dv_buf

    return dk_local, dv_local


ring_forward = build_ring_forward(RingAttentionConfig())
