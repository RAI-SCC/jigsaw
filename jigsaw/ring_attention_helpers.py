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


@torch.no_grad()
def ring_forward_sendX(
    Q,
    X,
    W_qkv,
    scale,
    group,
    empty,
    zeros,
    rank,
    world_size,
    send_X_or_KV=0,
    save_X_or_KV=0,
    save_local_or_global=0,
    offloader=None,
):
    # if send_X_or_KV == 1:
    #     assert save_X_or_KV == 0, "Cannot save X if we are sending KV"
    B, H, N_local, d = Q.shape
    world_size = dist.get_world_size(group)
    O = torch.zeros_like(Q)
    l = torch.full((B, H, N_local), float("-inf"), device=Q.device, dtype=Q.dtype)

    Xj = X
    chunks_to_save = []  # either X or KV

    # Calculate local QKV
    KV = (
        (Xj @ W_qkv[d * H :, :].mT).reshape(B, N_local, 2, H, d).permute(2, 0, 3, 1, 4)
    ).contiguous()

    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
    # Kick off first send before any compute
    
    match send_X_or_KV:
        case 0:
            recv_buf = torch.zeros_like(X)  # x or KV
            recv_buf, handle = _ring_send_recv(
                Xj, world_size, rank, group, empty, recv_buf
            )
        case 1:
            recv_buf = torch.zeros_like(KV)  # x or KV
            recv_buf, handle = _ring_send_recv(
                KV, world_size, rank, group, empty, recv_buf
            )

    for step in range(world_size):
        if step > 0:
            # Recompute KV from current Xj while communication is in flight
            if send_X_or_KV == 0:
                KV = (
                    (Xj @ W_qkv[d * H :, :].mT)
                    .reshape(B, N_local, 2, H, d)
                    .permute(2, 0, 3, 1, 4)
                )
                Kj = KV[0, :].contiguous()
                Vj = KV[1, :].contiguous()

        # store things for the backward pass
        if save_local_or_global == 0:  # save locally only
            # only save on the first pass
            if step == 0:
                if save_X_or_KV == 0:
                    Xj_cpu = offloader.offload(Xj)
                    chunks_to_save.append(Xj_cpu)
                    torch.cuda.synchronize()
                    print(torch.cuda.memory_allocated() / 1e9, "GB before")
                    del Xj
                    torch.cuda.empty_cache()
                    print(torch.cuda.memory_allocated() / 1e9, "GB after")
                elif save_X_or_KV == 1:
                    KV = torch.concat([Kj, Vj], dim=0)
                    KV_cpu = offloader.offload(KV)
                    chunks_to_save.append(KV_cpu)
                    print(torch.cuda.memory_allocated() / 1e9, "GB before")
                    torch.cuda.synchronize()
                    del KV
                    torch.cuda.empty_cache()
                    print(torch.cuda.memory_allocated() / 1e9, "GB before")
        else:
            if save_X_or_KV == 0:
                if offloader is not None:
                    print(torch.cuda.memory_allocated() / 1e9, "GB before")
                    Xj_cpu = offloader.offload(Xj)
                    chunks_to_save.append(Xj_cpu)
                    torch.cuda.synchronize()
                    del Xj
                    torch.cuda.empty_cache()
                    print(torch.cuda.memory_allocated() / 1e9, "GB after")
                else:
                    chunks_to_save.append(Xj) # save X 
            else:
                KV = torch.concat([Kj, Vj], dim=0)
                if offloader is not None:
                    KV_cpu = offloader.offload(KV).contiguous()
                    chunks_to_save.append(KV_cpu)
                    print(torch.cuda.memory_allocated() / 1e9, "GB before")
                    del KV
                    torch.cuda.empty_cache()
                    print(torch.cuda.memory_allocated() / 1e9, "GB after")
                else:
                    chunks_to_save.append(KV)
        
        # compute flash attention on local chunk of data
        O_j, l_j, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
            Q,
            Kj,
            Vj,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )

        O, l = _online_softmax_merge(O, l, O_j, l_j)
        print(torch.cuda.memory_allocated() / 1e9, "GB before lj")
        del l_j, O_j
        print(torch.cuda.memory_allocated() / 1e9, "GB after lj")
        # for all but the last step, send stuff
        if step < world_size - 1:
            handle.wait()
            if send_X_or_KV == 0:
                Xj = recv_buf.clone()
                recv_buf2 = torch.zeros_like(Xj)  # fresh buffer for next recv
                recv_buf, handle = _ring_send_recv(
                    Xj, world_size, rank, group, empty, recv_buf2
                )
            else:
                KV = recv_buf.clone()
                Kj, Vj = recv_buf[0], recv_buf[1]
                recv_buf2 = torch.zeros_like(KV)  # fresh buffer for next recv
                recv_buf, handle = _ring_send_recv(
                    KV, world_size, rank, group, empty, recv_buf2
                )

            recv_buf = recv_buf2

    O = O.to(torch.bfloat16)

    return O, l, chunks_to_save  # and all the other things that need to be saved

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
    elif save_X_or_KV == 1:
        KV_chunks = chunks_to_save

    B, H, Nlocal, d = Q.shape
    dQ = torch.zeros_like(Q, dtype=torch.float32)

    # We need to rotate dK, dV counter-clockwise back to home ranks
    # At step s in forward, KV came from rank (rank - s) % world_size
    # So dKj/dVj must go to rank (rank - s) % world_size
    # We do this by rotating dKj/dVj one step counter-clockwise per step

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

    for step in range(world_size):
        print(f'step {step}')
        if save_local_or_global == 1:
            if save_X_or_KV == 1:
                if offloader is not None:
                    KV_gpu = offloader.prefetch(KV_chunks[step])
                    offloader.sync()
                    KV = KV_gpu
                else:
                    KV = KV_chunks[step]
                Kj, Vj = torch.chunk(KV, 2, dim=0)
                Kj, Vj = Kj.contiguous(), Vj.contiguous()
            else: # save X or KV = 0
                if offloader is not None:
                    Xj = offloader.prefetch(X_chunks[step])
                    offloader.sync()
                else:
                    Xj = X_chunks[step]
                KV = (
                    (Xj @ W_qkv[d * H :, :].mT)
                    .reshape(B, Nlocal, 2, H, d)
                    .permute(2, 0, 3, 1, 4)
                ).contiguous()
                Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
        elif save_local_or_global == 0:
            if step == 0:
                if save_X_or_KV == 1:
                    KV = KV_chunks[step]
                    Kj, Vj = torch.chunk(KV, 2, dim=0)
                    Kj, Vj = Kj.contiguous(), Vj.contiguous()
                else: # save X or KV = 0
                    Xj = X_chunks[step]
                    KV = (
                        (Xj @ W_qkv[d * H :, :].mT)
                        .reshape(B, Nlocal, 2, H, d)
                        .permute(2, 0, 3, 1, 4)
                    ).contiguous()
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                    
                # and send to the next rank
                if send_X_or_KV == 1:
                    # start here
                    recv_buf = torch.zeros_like(KV)  # fresh buffer for next recv
                    recv_buf, KV_handle = _ring_send_recv(
                        KV, world_size, rank, group, empty, recv_buf
                    )
                else:
                    recv_buf = torch.zeros_like(X)
                    recv_buf, X_handle = _ring_send_recv(
                        X, world_size, rank, group, empty, recv_buf
                    )
            elif step < world_size - 1:
                if send_X_or_KV == 0:
                    X_handle.wait()
                    Xj = recv_buf
                    KV = (
                        (Xj @ W_qkv[d * H :, :].mT)
                        .reshape(B, Nlocal, 2, H, d)
                        .permute(2, 0, 3, 1, 4)
                    )
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                    recv_buf2 = torch.zeros_like(Xj)  # fresh buffer for next recv
                    recv_buf, X_handle = _ring_send_recv(
                        Xj, world_size, rank, group, empty, recv_buf2
                    )
                else:
                    KV_handle.wait()
                    
                    Kj, Vj = torch.chunk(recv_buf, 2, dim=0)
                    KV = recv_buf # TODO: can I delete this?
                    recv_buf = torch.zeros_like(KV)  # fresh buffer for next recv
                    
                    recv_buf, KV_handle = _ring_send_recv(
                        KV, world_size, rank, group, empty, recv_buf
                    )
            else: # on the last step
                if send_X_or_KV == 0:
                    X_handle.wait()
                    Xj = recv_buf
                    KV = (
                        (Xj @ W_qkv[d * H :, :].mT)
                        .reshape(B, Nlocal, 2, H, d)
                        .permute(2, 0, 3, 1, 4)
                    )
                    Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
                else:
                    KV_handle.wait()
                    KV = recv_buf
                    Kj, Vj = torch.chunk(KV, 2, dim=0)
        
        print(f"step {step}", torch.cuda.memory_allocated() / 1e9, "GB")
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
        print(f'step backward flashd {step}')
        
        dQ += dQ_j

        # accumulate into the slice corresponding to peer's K chunk
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
            if step > 0:
                dkvj_to_send = torch.concat([dKj, dVj], dim=0).contiguous()
                recv_buffer = torch.empty_like(
                    dkvj_to_send, device=dkvj_to_send.device, dtype=dkvj_to_send.dtype
                )
                recv_buffer, handle = _ring_send_recv(
                    dkvj_to_send,
                    world_size=world_size,
                    rank=rank,
                    group=group,
                    empty=empty,
                    recv_buf=recv_buffer,
                    offset=step,
                    forward_or_backward=1
                )
                handle.wait()
                dkj, dvj = torch.chunk(recv_buffer, 2, dim=0)
                dK_buf += dkj.clone()
                dV_buf += dvj.clone()
                print(torch.cuda.memory_allocated() / 1e9, "GB before dkvj")
                torch.cuda.synchronize()
                del dkj, dvj, recv_buffer
                torch.cuda.empty_cache()
                print(torch.cuda.memory_allocated() / 1e9, "GB after dkvj")
                
            else:
                dK_buf += dKj # accumulate locally
                dV_buf += dVj # accumulate locally
        

    if send_dKV_or_not == 1:
        # after loop: all-reduce across ranks to sum Q contributions
        dist.all_reduce(dK_buf, op=dist.ReduceOp.SUM, group=group)
        dist.all_reduce(dV_buf, op=dist.ReduceOp.SUM, group=group)

        # then slice out local portion if needed
        dK_local = dK_buf[:, :, rank * Nlocal : (rank + 1) * Nlocal]
        dV_local = dV_buf[:, :, rank * Nlocal : (rank + 1) * Nlocal]
    else:
        dK_local = dK_buf
        dV_local = dV_buf
    return dQ, dK_local, dV_local


# code saved
# recv_dK = torch.empty_like(Q)
# recv_dV = torch.empty_like(Q)

# if save_local_or_global == 0:
#     if save_X_or_KV == 0:
#         recv_buf = torch.zeros_like(X)
#         # Do I need to send Xj backward?
#         recv_buf, handle = _ring_send_recv(X, world_size, rank, group, empty, recv_buf)
#     if save_X_or_KV == 1:
#         recv_buf = torch.zeros_like(
#             KV_chunks[0]
#         )  # already have a local KV. Syntax may vary wrt. dimension.
#         # Do I need to send Xj backward?
#         KV = KV_chunks[0].contiguous()
#         Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
#         recv_buf, handle = _ring_send_recv(KV, world_size, rank, group, empty, recv_buf)
#         print(f'Step {step} KV sum is {KV.sum()}')

# for step in range()

# if we saved X, need to recompute KV
# if save_X_or_KV == 0 and save_local_or_global == 1:
#     Xj = X_chunks[step]
#     KV = (
#         (Xj @ W_qkv[d * H :, :].mT)
#         .reshape(B, Nlocal, 2, H, d)
#         .permute(2, 0, 1, 3, 4)
#     )
#     Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
# elif save_X_or_KV == 1 and save_local_or_global == 1:
#     Kj, Vj = KV_chunks[step]
#     Kj, Vj = Kj.contiguous(), Vj.contiguous()
#     # print(f'step {step} Kj sum {Kj.sum()}')
# elif save_X_or_KV == 0 and save_local_or_global == 0 and step > 0:
#     # some communication would be required here
#     Xj = X_chunks[step]
#     KV = (
#         (Xj @ W_qkv[d * H :, :].mT)
#         .reshape(B, Nlocal, 2, H, d)
#         .permute(2, 0, 1, 3, 4)
#     )
# elif save_X_or_KV == 1 and save_local_or_global == 0:


# print(f'R{rank}/{world_size}: dK_local sum {dK_local.sum()}')

# else:
#     dK_buf += dKj
#     dV_buf += dVj
# else:
#     send_to   = (rank - step) % world_size # owner of K I used
#     recv_from = (rank + step) % world_size # who used my K

#     recv_dK = torch.empty_like(dK_buf)
#     recv_dV = torch.empty_like(dV_buf)
#     reqs = dist.batch_isend_irecv([
#         dist.P2POp(dist.isend, dKj.contiguous(), send_to, group),
#         dist.P2POp(dist.isend, dVj.contiguous(), send_to, group),
#         dist.P2POp(dist.irecv, recv_dK,         recv_from, group),
#         dist.P2POp(dist.irecv, recv_dV,         recv_from, group),
#     ])
#     for r in reqs:
#         r.wait()
#     dK_buf += recv_dK
#     dV_buf += recv_dV

# Continue to send stuff backwards.
# if save_local_or_global == 0:
#     if save_X_or_KV == 0:
#         recv_buf = torch.zeros_like(Xj)
#         # Do I need to send Xj backward?
#         recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty, recv_buf)
#     if save_X_or_KV == 1:
#         recv_buf = torch.zeros_like(
#             KV_chunks[0]
#         )  # already have a local KV. Syntax may vary wrt. dimension.
#         # Do I need to send Xj backward?
#         recv_buf, handle = _ring_send_recv(KV, world_size, rank, group, empty, recv_buf)
