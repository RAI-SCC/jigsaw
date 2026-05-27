from enum import Enum

import torch
import torch.distributed as dist

from jigsaw.offload import NVLinkActivationOffloader


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


class RingAttentionConfig:
    """Config class for ring attention."""

    def __init__(
        self,
        send_in_forward: SendForward = SendForward.KV,
        save_for_backward: SaveBackward = SaveBackward.KV,
        save_scope: SaveScope = SaveScope.LOCAL,
        send_backward: SendBackward = SendBackward.SEND_DKV,
        offload: Offload = Offload.DISABLED,
        device: torch.device = None,
    ):
        self.send_in_forward = send_in_forward
        self.save_for_backward = save_for_backward
        self.save_scope = save_scope
        self.send_backward = send_backward
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
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"offload={self.offload.name})"
        )


@torch.no_grad()
def ring_forward(
    q, x, w_qkv, scale, group, empty, rank, world_size, ring_config: RingAttentionConfig
):
    """Forward pass for ring attention."""
    b, heads, n_local, d = q.shape
    world_size = dist.get_world_size(group)

    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x
    chunks_to_save = []

    # Local KV
    kv, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

    # Kick off first send
    # recv_buf either holds x or kv
    recv_buf, handle = send_in_forward(
        ring_config, x, kv, world_size, rank, group, empty
    )

    for step in range(world_size):
        if (step > 0) and (ring_config.send_in_forward == SendForward.X):
            # Recompute KV from current Xj (received in previous iter)
            kv, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk
        should_save = (ring_config.save_scope == SaveScope.GLOBAL) or (step == 0)

        if should_save:
            # append what we need to save for backward.
            # this can be x, kv, or the pointer to cpu offloaded copies of these
            chunks_to_save.append(
                save_for_backward(ring_config, chunks_to_save, x_j, kv)
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
            match ring_config.send_in_forward:
                case SendForward.X:
                    x_j = recv_buf
                case SendForward.KV:
                    kv = recv_buf
                    kj = kv[0]
                    vj = kv[1]
            
            # send x or k/v to the next rank
            next_recv = torch.empty_like(recv_buf)
            recv_buf, handle = _ring_send_recv(
                recv_buf, world_size, rank, group, empty, next_recv
            )

    # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
    if ring_config.offloader is not None:
        ring_config.offloader.wait_offloads()

    out = out.to(torch.bfloat16)
    return out, lse, chunks_to_save


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


def send_in_forward(ring_config, x, kv, world_size, rank, group, empty):
    # Send either X or KV in the forward pass
    match ring_config.send_in_forward:
        case SendForward.X:
            recv_buf = torch.empty_like(x)
            recv_buf, handle = _ring_send_recv(
                x, world_size, rank, group, empty, recv_buf
            )
        case SendForward.KV:
            recv_buf = torch.empty_like(kv)
            recv_buf, handle = _ring_send_recv(
                kv, world_size, rank, group, empty, recv_buf
            )

    return recv_buf, handle


def save_for_backward(ring_config, chunks_to_save, x, kv):
    match ring_config.save_for_backward:
        case SaveBackward.X:
            # offload X
            if ring_config.offloader is not None:
                xj_cpu = ring_config.offloader.offload(x)
                return xj_cpu
            else:
                # keep on GPU — must clone if Xj will be overwritten by recv next iter
                if ring_config.save_scope == SaveScope.GLOBAL:
                    return x.clone()
                else:
                    return x
        case SaveBackward.KV:
            # offload KV (note: KV is already a contiguous tensor of shape [2,B,H,N,d])
            if ring_config.offloader is not None:
                kv_cpu = ring_config.offloader.offload(kv)
                del kv
                return kv_cpu
            else:
                if ring_config.save_scope == SaveScope.GLOBAL:
                    return kv.clone()
                else:
                    return kv
    return chunks_to_save

def get_dkv_buffer(ring_config, q, world_size):
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
    # Handles how locally computed dk_j, dv_j is accumulated into the dK and dV buffer
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
