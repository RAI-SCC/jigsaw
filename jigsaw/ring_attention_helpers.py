import torch
import torch.distributed as dist
import sys
import gc


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


def _online_softmax_merge(O1, L1, O2, L2):
    L_new = torch.logaddexp(L1, L2)
    e1 = torch.exp(L1 - L_new)
    e2 = torch.exp(L2 - L_new)
    O_new = e1.unsqueeze(-1) * O1 + e2.unsqueeze(-1) * O2
    return O_new, L_new


def _ring_send_recv(
    tensor,
    world_size,
    rank,
    group,
    empty,
    recv_buf,
    forward_or_backward=0,
    offset=1,
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


@torch.no_grad()
def ring_forward_sendX(
    Q,
    X,
    W_qkv,
    scale,
    group,
    empty,
    rank,
    world_size,
    send_X_or_KV=0,
    save_X_or_KV=0,
    save_local_or_global=0,
    offloader=None,
):
    B, H, N_local, d = Q.shape
    world_size = dist.get_world_size(group)

    O = torch.zeros_like(Q)
    l = torch.full((B, H, N_local), float("-inf"), device=Q.device, dtype=Q.dtype)

    Xj = X
    chunks_to_save = []

    # Local KV
    KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, N_local, H, d)

    # Kick off first send
    if send_X_or_KV == 0:
        recv_buf = torch.empty_like(X)
        recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty, recv_buf)
    else:
        recv_buf = torch.empty_like(KV)
        recv_buf, handle = _ring_send_recv(KV, world_size, rank, group, empty, recv_buf)

    for step in range(world_size):
        if step > 0 and send_X_or_KV == 0:
            # Recompute KV from current Xj (received in previous iter)
            KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, N_local, H, d)

        # ---- save for backward ----
        # save_local_or_global == 0: only save own (step 0) chunk; backward re-rings
        # save_local_or_global == 1: save every step's chunk
        should_save = (save_local_or_global == 1) or (step == 0)

        if should_save:
            if save_X_or_KV == 0:
                # offload X
                if offloader is not None:
                    Xj_cpu = offloader.offload(Xj)
                    chunks_to_save.append(Xj_cpu)
                else:
                    # keep on GPU — must clone if Xj will be overwritten by recv next iter
                    chunks_to_save.append(
                        Xj.clone() if save_local_or_global == 1 else Xj
                    )
            else:
                # offload KV (note: KV is already a contiguous tensor of shape [2,B,H,N,d])
                if offloader is not None:
                    print(f"Memory before KV", torch.cuda.memory_allocated() / 1e9)
                    KV_cpu = offloader.offload(KV)
                    chunks_to_save.append(KV_cpu)
                    del KV
                    print(f"Memory after KV", torch.cuda.memory_allocated() / 1e9)
                else:
                    chunks_to_save.append(
                        KV.clone() if save_local_or_global == 1 else KV
                    )

        # ---- compute flash attention on local Q vs current Kj, Vj ----
        # Kj, Vj are views into KV; flash attention requires contiguous, so materialize.
        Kj_c = Kj.contiguous()
        Vj_c = Vj.contiguous()
        O_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            Q,
            Kj_c,
            Vj_c,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )
        O, l = _online_softmax_merge(O, l, O_j, l_j)
        print(f"Memory before", torch.cuda.memory_allocated() / 1e9)
        del O_j, l_j, Kj_c, Vj_c
        print(f"Memory after", torch.cuda.memory_allocated() / 1e9)

        # ---- ring rotate ----
        if step < world_size - 1:
            handle.wait()
            if send_X_or_KV == 0:
                # Drop Xj ref before reassigning. Note: at this point any pending
                # offload may still be reading Xj on the d2h_stream; the offloader
                # holds its own ref so this is safe.
                next_recv = torch.empty_like(recv_buf)
                Xj = recv_buf  # take ownership of received buffer
                recv_buf, handle = _ring_send_recv(
                    Xj, world_size, rank, group, empty, next_recv
                )
            else:
                next_recv = torch.empty_like(recv_buf)
                KV = recv_buf
                Kj = KV[0]
                Vj = KV[1]
                recv_buf, handle = _ring_send_recv(
                    KV, world_size, rank, group, empty, next_recv
                )

    # Make sure all offloads complete and GPU refs to offloaded tensors are dropped
    if offloader is not None:
        offloader.wait_offloads()

    O = O.to(torch.bfloat16)
    return O, l, chunks_to_save


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
    world_size,
    chunks_to_save=[],
    send_X_or_KV=0,
    save_X_or_KV=0,
    save_local_or_global=0,
    send_dKV_or_not=1,
    offloader=None,
    empty=None,
):
    rank = dist.get_rank(group)

    if save_X_or_KV == 0:
        X_chunks = chunks_to_save
    else:
        KV_chunks = chunks_to_save

    B, H, Nlocal, d = Q.shape
    dQ = torch.zeros_like(Q, dtype=torch.float32)

    if send_dKV_or_not == 1:
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
    else:
        dK_buf = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)
        dV_buf = torch.zeros(Q.shape, device=Q.device, dtype=Q.dtype)

    # Pre-issue first prefetch (for save_local_or_global == 1) so H2D overlaps
    # with the compute of step 0.
    next_prefetch = None
    next_prefetch_event = None

    def _start_prefetch(idx):
        """Kick off async prefetch for chunk idx; returns (gpu_tensor, event) or None."""
        if save_local_or_global != 1 or offloader is None:
            return None, None
        if idx >= len(chunks_to_save):
            return None, None
        gpu, ev = offloader.prefetch(chunks_to_save[idx])
        return gpu, ev

    # Step 0 chunk: prefetch if applicable
    if save_local_or_global == 1 and offloader is not None:
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
        if save_local_or_global == 1:
            # we have all chunks saved; use the prefetched one
            if offloader is not None:
                offloader.wait_prefetch(cur_chunk_event)
                chunk = cur_chunk_gpu
                # kick off next prefetch ASAP to overlap with compute
                cur_chunk_gpu, cur_chunk_event = next_prefetch, next_prefetch_event
                next_prefetch, next_prefetch_event = _start_prefetch(step + 2)
            else:
                chunk = chunks_to_save[step]

            if save_X_or_KV == 1:
                KV = chunk
                Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
            else:
                Xj = chunk
                KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                Kj, Vj = Kj.contiguous(), Vj.contiguous()

        else:
            # save_local_or_global == 0: only own chunk saved; ring around
            if step == 0:
                if save_X_or_KV == 1:
                    if offloader is not None:
                        gpu, ev = offloader.prefetch(KV_chunks[0])
                        offloader.wait_prefetch(ev)
                        KV = gpu
                    else:
                        KV = KV_chunks[0]
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                else:
                    if offloader is not None:
                        gpu, ev = offloader.prefetch(X_chunks[0])
                        offloader.wait_prefetch(ev)
                        Xj = gpu
                    else:
                        Xj = X_chunks[0]
                    KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                    Kj, Vj = Kj.contiguous(), Vj.contiguous()

                # Kick off first ring send
                if send_X_or_KV == 0:
                    # we need an Xj to send. If we saved KV instead of X, we don't have it.
                    # In that case caller must have passed X.
                    to_send = Xj if save_X_or_KV == 0 else X
                    recv_buf = torch.empty_like(to_send)
                    recv_buf, X_handle = _ring_send_recv(
                        to_send, world_size, rank, group, empty, recv_buf
                    )
                else:
                    recv_buf = torch.empty_like(KV)
                    recv_buf, KV_handle = _ring_send_recv(
                        KV, world_size, rank, group, empty, recv_buf
                    )

            else:
                # step > 0: receive next chunk
                if send_X_or_KV == 0:
                    X_handle.wait()
                    Xj = recv_buf
                    KV, Kj, Vj = _compute_KV(Xj, W_qkv, B, Nlocal, H, d)
                    Kj, Vj = Kj.contiguous(), Vj.contiguous()
                    if step < world_size - 1:
                        next_recv = torch.empty_like(Xj)
                        recv_buf, X_handle = _ring_send_recv(
                            Xj, world_size, rank, group, empty, next_recv
                        )
                else:
                    KV_handle.wait()
                    KV = recv_buf
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                    if step < world_size - 1:
                        next_recv = torch.empty_like(KV)
                        recv_buf, KV_handle = _ring_send_recv(
                            KV, world_size, rank, group, empty, next_recv
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
        print(f"step {step} backwards before", torch.cuda.memory_allocated() / 1e9)
        del dQ_j, Kj, Vj, KV

        if save_local_or_global == 1 and save_X_or_KV == 0:
            del Xj  # was loaded from CPU; can drop now
        print(f"step {step} backwards after", torch.cuda.memory_allocated() / 1e9)

        # ---- accumulate dK, dV ----
        if send_dKV_or_not == 1:
            global_chunk_idx = (rank - step) % world_size
            dK_buf[
                :, :, global_chunk_idx * Nlocal : (global_chunk_idx + 1) * Nlocal
            ] += dKj
            dV_buf[
                :, :, global_chunk_idx * Nlocal : (global_chunk_idx + 1) * Nlocal
            ] += dVj
            del dKj, dVj
        else:
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
                    rank=rank,
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

    if send_dKV_or_not == 1:
        dist.all_reduce(dK_buf, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dV_buf, op=dist.ReduceOp.SUM, group=group)
        dK_local = dK_buf[:, :, rank * Nlocal : (rank + 1) * Nlocal]
        dV_local = dV_buf[:, :, rank * Nlocal : (rank + 1) * Nlocal]
    else:
        dK_local = dK_buf
        dV_local = dV_buf

    return dQ, dK_local, dV_local
