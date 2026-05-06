import torch
import torch.distributed as dist

def _online_softmax_merge(O1, L1, O2, L2):
    L_new = torch.logaddexp(L1, L2)
    e1 = torch.exp(L1 - L_new)
    e2 = torch.exp(L2 - L_new)
    O_new = e1.unsqueeze(-1) * O1 + e2.unsqueeze(-1) * O2
    return O_new, L_new


def _ring_send_recv(tensor, world_size, rank, group, empty, recv_buf, forward_or_backward=0):
    if forward_or_backward == 0:
        send_to = (rank + 1) % world_size
        recv_from = (rank - 1) % world_size
    else:
        send_to = (rank - 1) % world_size
        recv_from = (rank + 1) % world_size
    output_list = [recv_buf if i == recv_from else empty for i in range(world_size)]
    input_list = [tensor if i == send_to else empty for i in range(world_size)]
    handle = dist.all_to_all(output_list, input_list, group=group, async_op=True)
    return recv_buf, handle


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
    offload=False,
):

    B, H, N_local, d = Q.shape
    world_size = dist.get_world_size(group)
    O = torch.zeros_like(Q)
    l = torch.full((B, H, N_local), float("-inf"), device=Q.device, dtype=Q.dtype)

    Xj = X.clone()
    chunks_to_save = []  # either X or KV

    Xj = X.clone()
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
                QKVj = (
                    (Xj @ W_qkv[d * H :, :].mT)
                    .reshape(B, N_local, 2, H, d)
                    .permute(2, 0, 3, 1, 4)
                )
                Kj = QKVj[0, :].contiguous()
                Vj = QKVj[1, :].contiguous()
                KV = QKVj.contiguous()
        
        # store things for the backward pass
        if save_local_or_global == 0:  # save locally only
            # only save on the first pass
            if step == 0:
                if save_X_or_KV == 0:
                    chunks_to_save.append(Xj)
                elif save_X_or_KV == 1:
                    chunks_to_save.append([Kj, Vj])
        else:
            if save_X_or_KV == 0:
                chunks_to_save.append(Xj)
            else:
                chunks_to_save.append([Kj, Vj])

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
    offload=False,
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

    dK_buf = torch.zeros([*Q.shape[0:2], world_size*Q.shape[2], *Q.shape[3:]], device=Q.device, dtype=Q.dtype)
    dV_buf = torch.zeros([*Q.shape[0:2], world_size*Q.shape[2], *Q.shape[3:]], device=Q.device, dtype=Q.dtype)
    
    for step in range(world_size):
        peer = step  # which chunk of global K this is

        KV = KV_chunks[step]
        Kj, Vj = KV[0].contiguous(), KV[1].contiguous()
        
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
        
        dQ = dQ + dQ_j
        
        # accumulate into the slice corresponding to peer's K chunk
        global_chunk_idx = (rank - step) % world_size
        dK_buf[:, :, global_chunk_idx*Nlocal:(global_chunk_idx+1)*Nlocal] += dKj
        dV_buf[:, :, global_chunk_idx*Nlocal:(global_chunk_idx+1)*Nlocal] += dVj
    

    # after loop: all-reduce across ranks to sum Q contributions
    dist.all_reduce(dK_buf, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(dV_buf, op=dist.ReduceOp.SUM, group=group)
    
    # then slice out local portion if needed
    dK_local = dK_buf[:, :, rank*Nlocal:(rank+1)*Nlocal]
    dV_local = dV_buf[:, :, rank*Nlocal:(rank+1)*Nlocal]

    return dQ, dK_local, dV_local

# def ring_backward(
#     dO,
#     Q,
#     K,
#     V,
#     O,
#     L,
#     N,
#     scale,
#     group,
#     world_size
# ):
#     rank = dist.get_rank(group)

#     dQ = torch.zeros_like(Q, dtype=torch.float32)
    
#     # We need to rotate dK, dV counter-clockwise back to home ranks
#     # At step s in forward, KV came from rank (rank - s) % world_size
#     # So dKj/dVj must go to rank (rank - s) % world_size
#     # We do this by rotating dKj/dVj one step counter-clockwise per step

#     dK_buf = torch.zeros([*K.shape[0:2], world_size*K.shape[2], *K.shape[3:]], device=K.device, dtype=K.dtype)
#     dV_buf = torch.zeros([*V.shape[0:2], world_size*V.shape[2], *V.shape[3:]], device=V.device, dtype=V.dtype)
    
#     for step in range(world_size):
#         peer = (step) % world_size  # which chunk of global K this is
        
#         dQ_j, dKj, dVj = torch.ops.aten._scaled_dot_product_flash_attention_backward(
#             dO.contiguous(),
#             Q.contiguous(),
#             K.contiguous(),
#             V.contiguous(),
#             O.contiguous(),
#             L.contiguous(),
#             dropout_p=0.0,
#             is_causal=False,
#             scale=scale,
#             cum_seq_q=None,
#             cum_seq_k=None,
#             max_q=N,
#             max_k=N,
#             philox_seed=torch.zeros(1, dtype=torch.long, device=Q.device),
#             philox_offset=torch.zeros(1, dtype=torch.long, device=Q.device),
#         )
        
#         dQ += dQ_j
        
#         # accumulate into the slice corresponding to peer's K chunk
#         global_chunk_idx = (rank - step) % world_size
#         # dK_buf[:, global_chunk_idx*Nlocal:(global_chunk_idx+1)*Nlocal] += dKj
#         # dV_buf[:, global_chunk_idx*Nlocal:(global_chunk_idx+1)*Nlocal] += dVj
#         if rank == 0:
#             print(f'R {rank} step {step} dQ {dQ_j}')
#         # print(f'R {rank} dK, dV sum {dKj.sum()}, {dVj.sum()}')

#     # after loop: all-reduce across ranks to sum Q contributions
#     dist.all_reduce(dK_buf, op=dist.ReduceOp.SUM, group=group)
#     dist.all_reduce(dV_buf, op=dist.ReduceOp.SUM, group=group)
    
#     # then slice out local portion if needed
#     dK_local = dK_buf#[:, rank*Nlocal:(rank+1)*Nlocal]
#     dV_local = dV_buf#[:, rank*Nlocal:(rank+1)*Nlocal]

#     return dQ, dK_local, dV_local

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
