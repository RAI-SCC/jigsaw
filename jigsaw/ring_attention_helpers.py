from enum import Enum

import torch
import torch.distributed as dist
from torch.cuda import nvtx

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
    nvtx.range_push("Local KV")
    kv, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
    nvtx.range_pop()

    # Kick off first send
    nvtx.range_push("First Ring Send")
    match ring_config.send_in_forward:
        case SendForward.X:
            recv_buf = torch.empty_like(x)
            recv_buf, handle = _ring_send_recv(
                x_j, world_size, rank, group, empty, recv_buf
            )
        case SendForward.KV:
            recv_buf = torch.empty_like(kv)
            recv_buf, handle = _ring_send_recv(
                kv, world_size, rank, group, empty, recv_buf
            )
    nvtx.range_pop()

    for step in range(world_size):
        if (step > 0) and (ring_config.send_in_forward == SendForward.X):
            # Recompute KV from current Xj (received in previous iter)
            nvtx.range_push("Compute KV")
            kv, kj, vj = _compute_kv(x_j, w_qkv, b, n_local, heads, d)
            nvtx.range_pop()

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk
        should_save = (ring_config.save_scope == SaveScope.GLOBAL) or (step == 0)

        if should_save:
            nvtx.range_push("Save for Backward")
            match ring_config.save_for_backward:
                case SaveBackward.X:
                    # offload X
                    if ring_config.offloader is not None:
                        xj_cpu = ring_config.offloader.offload(x_j)
                        chunks_to_save.append(xj_cpu)
                    else:
                        # keep on GPU — must clone if Xj will be overwritten by recv next iter
                        chunks_to_save.append(
                            x_j.clone()
                            if (ring_config.save_scope == SaveScope.GLOBAL)
                            else x_j
                        )
                case SaveBackward.KV:
                    # offload KV (note: KV is already a contiguous tensor of shape [2,B,H,N,d])
                    if ring_config.offloader is not None:
                        kv_cpu = ring_config.offloader.offload(kv)
                        chunks_to_save.append(kv_cpu)
                        del kv
                    else:
                        chunks_to_save.append(
                            kv.clone()
                            if (ring_config.save_scope == SaveScope.GLOBAL)
                            else kv
                        )
            nvtx.range_pop()

        # ---- compute flash attention on local Q vs current Kj, Vj ----
        # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
        nvtx.range_push("Contiguous KV")
        kj_c = kj.contiguous()
        vj_c = vj.contiguous()
        nvtx.range_pop()
        nvtx.range_push("SDPFA")
        out_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q,
            kj_c,
            vj_c,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )
        nvtx.range_pop()
        nvtx.range_push("Online Softmax")
        out, lse = _online_softmax_merge(out, lse, out_j, l_j)
        nvtx.range_pop()
        del out_j, l_j, kj_c, vj_c

        # ---- ring rotate ----
        if step < world_size - 1:
            handle.wait()
            nvtx.range_push("Ring Rotate")
            match ring_config.send_in_forward:
                case SendForward.X:
                    # Drop Xj ref before reassigning. Note: at this point any pending
                    # offload may still be reading Xj on the d2h_stream; the offloader
                    # holds its own ref so this is safe.
                    next_recv = torch.empty_like(recv_buf)
                    x_j = recv_buf  # take ownership of received buffer
                    recv_buf, handle = _ring_send_recv(
                        x_j, world_size, rank, group, empty, next_recv
                    )
                case SendForward.KV:
                    next_recv = torch.empty_like(recv_buf)
                    kv = recv_buf
                    kj = kv[0]
                    vj = kv[1]
                    recv_buf, handle = _ring_send_recv(
                        kv, world_size, rank, group, empty, next_recv
                    )
            nvtx.range_pop()

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

    b, h, n_local, d = q.shape
    dq = torch.zeros_like(q, dtype=torch.float32)

    # Get buffers
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
    nvtx.range_push("Prefetch Step 0 Chunk")
    if ring_config.save_scope == SaveScope.GLOBAL and ring_config.offloader is not None:
        cur_chunk_gpu, cur_chunk_event = _start_prefetch(0)
        next_prefetch, next_prefetch_event = _start_prefetch(1)
    else:
        cur_chunk_gpu, cur_chunk_event = None, None
    nvtx.range_pop()

    # State for ring rotation
    recv_buf = None
    x_handle = None
    kv_handle = None

    for step in range(world_size):
        # ---- obtain Kj, Vj for this step ----
        match ring_config.save_scope:
            case SaveScope.GLOBAL:
                nvtx.range_push("Global Save Backward")
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
                    kv, kj, vj = _compute_kv(xj, w_qkv, b, n_local, h, d)
                    kj, vj = kj.contiguous(), vj.contiguous()
                nvtx.range_pop()
            case SaveScope.LOCAL:
                # save_local_or_global == 0: only own chunk saved; ring around
                nvtx.range_push("Local Save Backward")
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
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, n_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()

                    # Kick off first ring send
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            # we need an Xj to send. If we saved KV instead of X, we don't have it.
                            # In that case caller must have passed X.
                            to_send = (
                                xj
                                if (ring_config.save_for_backward == SaveBackward.X)
                                else x
                            )
                            recv_buf = torch.empty_like(to_send)
                            recv_buf, x_handle = _ring_send_recv(
                                to_send, world_size, local_rank, group, empty, recv_buf
                            )
                        case SendForward.KV:
                            recv_buf = torch.empty_like(kv)
                            recv_buf, kv_handle = _ring_send_recv(
                                kv, world_size, local_rank, group, empty, recv_buf
                            )

                else:
                    # step > 0: receive next chunk
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            x_handle.wait()
                            xj = recv_buf
                            kv, kj, vj = _compute_kv(xj, w_qkv, b, n_local, h, d)
                            kj, vj = kj.contiguous(), vj.contiguous()
                            if step < world_size - 1:
                                next_recv = torch.empty_like(xj)
                                recv_buf, x_handle = _ring_send_recv(
                                    xj, world_size, local_rank, group, empty, next_recv
                                )
                        case SendForward.KV:
                            kv_handle.wait()
                            kv = recv_buf
                            kj, vj = kv[0].contiguous(), kv[1].contiguous()
                            if step < world_size - 1:
                                next_recv = torch.empty_like(kv)
                                recv_buf, kv_handle = _ring_send_recv(
                                    kv, world_size, local_rank, group, empty, next_recv
                                )
                nvtx.range_pop()

        # ---- flash attention backward ----
        nvtx.range_push("Flash Attention Backward")
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
            max_q=n_local,
            max_k=n_local,
            philox_seed=torch.zeros(1, dtype=torch.long, device=q.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=q.device),
        )
        dq += dq_j
        nvtx.range_pop()
        del dq_j, kj, vj, kv
        # ---- accumulate dK, dV ----
        nvtx.range_push("Accumulate dK, dV")
        match ring_config.send_backward:
            case SendBackward.GATHER_DKV:
                global_chunk_idx = (local_rank - step) % world_size
                dk_buf[
                    :, :, global_chunk_idx * n_local : (global_chunk_idx + 1) * n_local
                ] += dk_j
                dv_buf[
                    :, :, global_chunk_idx * n_local : (global_chunk_idx + 1) * n_local
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
        nvtx.range_pop()

    nvtx.range_push("Finalize Backward")
    match ring_config.send_backward:
        case SendBackward.GATHER_DKV:
            dist.all_reduce(dk_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dv_buf, op=dist.ReduceOp.SUM, group=group)
            dk_local = dk_buf[:, :, local_rank * n_local : (local_rank + 1) * n_local]
            dv_local = dv_buf[:, :, local_rank * n_local : (local_rank + 1) * n_local]
        case SendBackward.SEND_DKV:
            dk_local = dk_buf
            dv_local = dv_buf
    nvtx.range_pop()

    return dq, dk_local, dv_local

def get_dkv_buffer(ring_config, q, world_size):
    # If
    
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