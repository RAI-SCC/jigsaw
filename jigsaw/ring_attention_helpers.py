import torch
import torch.distributed as dist
from enum import Enum
from jigsaw.offload import NVLinkActivationOffloader


def _online_softmax_merge(O1, L1, O2, L2):
    L_new = torch.logaddexp(L1, L2)
    e1 = torch.exp(L1 - L_new)
    e2 = torch.exp(L2 - L_new)
    O_new = e1.unsqueeze(-1) * O1 + e2.unsqueeze(-1) * O2
    return O_new, L_new


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


def _compute_KV(Xj, W_qkv, B, N_local, H, d):
    """Compute (K, V) from X. Returns (KV, Kj, Vj). KV is the underlying buffer."""
    KV = (
        (Xj @ W_qkv[d * H :, :].mT).reshape(B, N_local, 2, H, d).permute(2, 0, 3, 1, 4)
    ).contiguous()
    Kj = KV[0]
    Vj = KV[1]
    return KV, Kj, Vj


# Module-level enum definitions
class SendForward(Enum):
    X = 0
    KV = 1


class SaveBackward(Enum):
    X = 0
    KV = 1


class SaveScope(Enum):
    LOCAL = 0
    GLOBAL = 1


class Offload(Enum):
    DISABLED = 0
    CPU = 1


class SendBackward(Enum):
    SEND_DKV = 0
    DISABLED = 1


class RingAttentionConfig:
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
        if offload == Offload.DISABLED:
            return None
        elif offload == Offload.CPU:
            return NVLinkActivationOffloader(device)

    def __repr__(self):
        return (
            f"RingAttentionConfig("
            f"send_in_forward={self.send_in_forward.name}, "
            f"save_for_backward={self.save_for_backward.name}, "
            f"save_scope={self.save_scope.name}, "
            f"offload={self.offload.name})"
        )


@torch.no_grad()
def ring_forward_sendX(
    q, x, w_qkv, scale, group, empty, rank, world_size, ring_config: RingAttentionConfig
):
    b, heads, n_local, d = q.shape
    world_size = dist.get_world_size(group)

    out = torch.zeros_like(q)
    lse = torch.full((b, heads, n_local), float("-inf"), device=q.device, dtype=q.dtype)

    x_j = x
    chunks_to_save = []

    # Local KV
    kv, kj, vj = _compute_KV(x_j, w_qkv, b, n_local, heads, d)

    # Kick off first send
    match ring_config.send_in_forward:
        case SendForward.X:
            recv_buf = torch.empty_like(x)
            recv_buf, handle = _ring_send_recv(
                x_j, world_size, rank, group, empty, recv_buf
            )
        case SendForward.KV:
            recv_buf = torch.empty_like(kv)
            recv_buf, handle = _ring_send_recv(kv, world_size, rank, group, empty, recv_buf)
            
    for step in range(world_size):
        if (step > 0) and (ring_config.send_in_forward == SendForward.X):
            # Recompute KV from current Xj (received in previous iter)
            kv, kj, vj = _compute_KV(x_j, w_qkv, b, n_local, heads, d)

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk
        should_save = (ring_config.save_scope == SaveScope.GLOBAL) or (step == 0)

        if should_save:
            match ring_config.save_for_backward:
                case SaveBackward.X:
                    # offload X
                    if ring_config.offloader is not None:
                        Xj_cpu = ring_config.offloader.offload(x_j)
                        chunks_to_save.append(Xj_cpu)
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
                        KV_cpu = ring_config.offloader.offload(kv)
                        chunks_to_save.append(KV_cpu)
                        del kv
                    else:
                        chunks_to_save.append(
                            kv.clone()
                            if (ring_config.save_scope == SaveScope.GLOBAL)
                            else kv
                        )

        # ---- compute flash attention on local Q vs current Kj, Vj ----
        # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
        Kj_c = kj.contiguous()
        Vj_c = vj.contiguous()
        O_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            q,
            Kj_c,
            Vj_c,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )
        out, lse = _online_softmax_merge(out, lse, O_j, l_j)
        del O_j, l_j, Kj_c, Vj_c

        # ---- ring rotate ----
        if step < world_size - 1:
            handle.wait()
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

    # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
    if ring_config.offloader is not None:
        ring_config.offloader.wait_offloads()

    out = out.to(torch.bfloat16)
    return out, lse, chunks_to_save


@torch.no_grad()
def ring_backward(
    dO,
    Q,
    O,
    X,
    L,
    W_qkv,
    scale,
    group,
    local_rank,
    world_size,
    chunks_to_save,
    ring_config: RingAttentionConfig,
    empty=None,
):
    if ring_config.save_for_backward == SaveBackward.X:
        X_chunks = chunks_to_save
    elif ring_config.save_for_backward == SaveBackward.KV:
        KV_chunks = chunks_to_save

    B, H, Nlocal, d = Q.shape
    dQ = torch.zeros_like(Q, dtype=torch.float32)

    match ring_config.send_backward:
        case SendBackward.DISABLED:  # send dK = 1
            dK_buf = torch.zeros(
                [*Q.shape[0:2], world_size * Q.shape[2], *Q.shape[3:]],
                device=Q.device,
                dtype=Q.dtype,
            )
            dV_buf = torch.zeros(
                [*Q.shape[0:2], world_size * Q.shape[2], *Q.shape[3:]],
                device=Q.device,
                dtype=Q.dtype,
            )
        case SendBackward.SEND_DKV:
            dK_buf = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)
            dV_buf = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)

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
    X_handle = None
    KV_handle = None

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
                    KV = chunk
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                elif ring_config.save_for_backward == SaveBackward.X:
                    Xj = chunk
                    KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                    Kj, Vj = Kj.contiguous(), Vj.contiguous()
            case SaveScope.LOCAL:
                # save_local_or_global == 0: only own chunk saved; ring around
                if step == 0:
                    match ring_config.save_for_backward:
                        case SaveBackward.KV:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(KV_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                KV = gpu
                            else:
                                KV = KV_chunks[0]
                            Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                        case SaveBackward.X:
                            if ring_config.offloader is not None:
                                gpu, ev = ring_config.offloader.prefetch(X_chunks[0])
                                ring_config.offloader.wait_prefetch(ev)
                                Xj = gpu
                            else:
                                Xj = X_chunks[0]
                            KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                            Kj, Vj = Kj.contiguous(), Vj.contiguous()

                    # Kick off first ring send
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            # we need an Xj to send. If we saved KV instead of X, we don't have it.
                            # In that case caller must have passed X.
                            to_send = (
                                Xj if (ring_config.save_for_backward == SaveBackward.X) else X
                            )
                            recv_buf = torch.empty_like(to_send)
                            recv_buf, X_handle = _ring_send_recv(
                                to_send, world_size, local_rank, group, empty, recv_buf
                            )
                        case SendForward.KV:
                            recv_buf = torch.empty_like(KV)
                            recv_buf, KV_handle = _ring_send_recv(
                                KV, world_size, local_rank, group, empty, recv_buf
                            )

                else:
                    # step > 0: receive next chunk
                    match ring_config.send_in_forward:
                        case SendForward.X:
                            X_handle.wait()
                            Xj = recv_buf
                            KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                            Kj, Vj = Kj.contiguous(), Vj.contiguous()
                            if step < world_size - 1:
                                next_recv = torch.empty_like(Xj)
                                recv_buf, X_handle = _ring_send_recv(
                                    Xj, world_size, local_rank, group, empty, next_recv
                                )
                        case SendForward.KV:
                            KV_handle.wait()
                            KV = recv_buf
                            Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                            if step < world_size - 1:
                                next_recv = torch.empty_like(KV)
                                recv_buf, KV_handle = _ring_send_recv(
                                    KV, world_size, local_rank, group, empty, next_recv
                                )

        # ---- flash attention backward ----
        dQ_j, dKj, dVj = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            dO,
            Q,
            Kj,
            Vj,
            O,
            L,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            philox_seed=torch.zeros(1, dtype=torch.long, device=Q.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=Q.device),
        )
        dQ += dQ_j
        del dQ_j, Kj, Vj, KV
        # ---- accumulate dK, dV ----
        match ring_config.send_backward:
            case SendBackward.DISABLED:
                global_chunk_idx = (local_rank - step) % world_size
                dK_buf[
                    :, :, global_chunk_idx * Nlocal : (global_chunk_idx + 1) * Nlocal
                ] += dKj
                dV_buf[
                    :, :, global_chunk_idx * Nlocal : (global_chunk_idx + 1) * Nlocal
                ] += dVj
                del dKj, dVj
            case SendBackward.SEND_DKV:
                if step == 0:
                    dK_buf += dKj
                    dV_buf += dVj
                    del dKj, dVj
                else:
                    dkv = torch.cat([dKj, dVj], dim=0).contiguous()
                    del dKj, dVj
                    recv_buffer = torch.empty_like(dkv)
                    recv_buffer, h = _ring_send_recv(
                        dkv,
                        world_size=world_size,
                        rank=local_rank,
                        group=group,
                        empty=empty,
                        recv_buf=recv_buffer,
                        offset=step,
                        forward_or_backward=1,
                    )
                    h.wait()
                    del dkv
                    dkj_r, dvj_r = torch.chunk(recv_buffer, 2, dim=0)
                    dK_buf += dkj_r
                    dV_buf += dvj_r
                    del dkj_r, dvj_r, recv_buffer

    match ring_config.send_backward:
        case SendBackward.DISABLED:
            dist.all_reduce(dK_buf, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(dV_buf, op=dist.ReduceOp.SUM, group=group)
            dK_local = dK_buf[:, :, local_rank * Nlocal : (local_rank + 1) * Nlocal]
            dV_local = dV_buf[:, :, local_rank * Nlocal : (local_rank + 1) * Nlocal]
        case SendBackward.SEND_DKV:
            dK_local = dK_buf
            dV_local = dV_buf

    return dQ, dK_local, dV_local
