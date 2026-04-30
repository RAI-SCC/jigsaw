import torch
import torch.distributed as dist

def _online_softmax_merge(O1, L1, O2, L2):
    L_new = torch.logaddexp(L1, L2)
    e1 = torch.exp(L1 - L_new)
    e2 = torch.exp(L2 - L_new)
    O_new = e1.unsqueeze(-1) * O1 + e2.unsqueeze(-1) * O2
    return O_new, L_new

def _ring_send_recv(tensor, world_size, rank, group, empty, recv_buf):
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size
    output_list = [recv_buf if i == recv_from else empty for i in range(world_size)]
    input_list = [tensor if i == send_to else empty for i in range(world_size)]
    handle = dist.all_to_all(output_list, input_list, group=group, async_op=True)
    return recv_buf, handle

def ring_forward_sendX(ctx, Q, X, W_qkv, scale, group, empty, zeros, rank, world_size, send_X_or_KV=0, save_X_or_KV=0, save_local_or_global=0, offload=False):
    send_X_or_KV=0, save_X_or_KV=0, save_local_or_global=0, offload=False
    
    B, N_local, H, d = Q.shape
    world_size = dist.get_world_size(group)
    O = torch.zeros_like(Q)
    l = torch.full((B, N_local, H), float("-inf"), device=Q.device, dtype=Q.dtype)

    Xj = X.clone()
    chunks_to_save = [] # either X or KV
        
    Xj = X.clone()
        
    # Calculate local QKV
    QKVj = (
        (Xj @ W_qkv[d * H :, :].mT)
        .reshape(B, N_local, 2, H, d)
        .permute(2, 0, 1, 3, 4)
    )
    
    # Kick off first send before any compute    
    match send_X_or_KV:
        case 0:
            recv_buf = torch.zeros_like(X) # x or KV
            recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty, recv_buf)
        case 1:
            KV = QKVj[1:]
            recv_buf = torch.zeros_like(KV) # x or KV
            recv_buf, handle = _ring_send_recv(KV, world_size, rank, group, empty, recv_buf)
    
    for step in range(world_size):
        if step > 0:
            # Recompute KV from current Xj while communication is in flight
            if send_X_or_KV == 0:
                QKVj = (
                    (Xj @ W_qkv[d * H :, :].mT)
                    .reshape(B, N_local, 2, H, d)
                    .permute(2, 0, 1, 3, 4)
                )
                Kj = QKVj[0, :].contiguous()
                Vj = QKVj[1, :].contiguous()       
        
        # store things for the backward pass
        if (save_local_or_global == 0): # save locally only
            # only save on the first pass
            if (step == 0): 
                if save_X_or_KV == 0:
                    chunks_to_save.append([Xj])
                else:
                    chunks_to_save.append([[Kj, Vj]])
        else:
            if save_X_or_KV == 0:
                chunks_to_save.append([Xj])
            else:
                chunks_to_save.append([[Kj, Vj]])
        
        # compute flash attention on local chunk of data
        O_j, l_j, *_ = torch.ops.aten._flash_attention_forward(
            Q,
            Kj,
            Vj,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=N_local,
            max_k=N_local,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )

        O, l = _online_softmax_merge(O, l, O_j, l_j.mT)

        # for all but the last step, send stuff
        if step < world_size - 1:
            handle.wait()
            if send_X_or_KV == 0:
                Xj = recv_buf.clone()
                recv_buf2 = torch.zeros_like(Xj)  # fresh buffer for next recv
                recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty, recv_buf2)
            else:
                Kj, Vj = torch.chunk(recv_buf, chunks=2, dim=-1)
                recv_buf2 = torch.zeros_like(KV)  # fresh buffer for next recv
                recv_buf, handle = _ring_send_recv(KV, world_size, rank, group, empty, recv_buf2)
            recv_buf = recv_buf2

    O = O.to(torch.bfloat16)
    
    return O # and all the other things that need to be saved

def backward(dO, Q, O, L, W_qkv, scale, group, world_size, chunks_to_save=[], send_X_or_KV=0, save_X_or_KV=0, save_local_or_global=0, offload=False):
        rank = dist.get_rank(group)

        if save_X_or_KV == 0:
            X_chunks = chunks_to_save
        elif save_X_or_KV == 1:
            KV_chunks = [(chunks_to_save[2 * i], chunks_to_save[2 * i + 1]) for i in range(world_size)]

        B, Nlocal, H, d = Q.shape

        dQ = torch.zeros_like(Q, dtype=Q.dtype)

        # We need to rotate dK, dV counter-clockwise back to home ranks
        # At step s in forward, KV came from rank (rank - s) % world_size
        # So dKj/dVj must go to rank (rank - s) % world_size
        # We do this by rotating dKj/dVj one step counter-clockwise per step

        dK = torch.zeros_like(Q, dtype=Q.dtype)
        dV = torch.zeros_like(Q, dtype=Q.dtype)
        
        if save_local_or_global == 1:
            if save_X_or_KV == 0:
                recv_buf = torch.zeros_like(Xj)
                # Do I need to send Xj backward?
                recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty)
            if save_X_or_KV == 1:
                recv_buf = torch.zeros_like(KV_chunks[0]) # already have a local KV. Syntax may vary wrt. dimension.
                # Do I need to send Xj backward?
                recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty)

        for step in range(world_size):
            home_rank = (rank - step) % world_size
            
            # if we saved X, need to recompute KV
            if save_X_or_KV == 0 and save_local_or_global == 1: 
                Xj = X_chunks[step]
                QKVj = (
                    (Xj @ W_qkv[d * h :, :].mT)
                    .reshape(B, Nlocal, 2, H, d)
                    .permute(2, 0, 1, 3, 4)
                )
                Kj, Vj = QKVj[1].contiguous(), QKVj[2].contiguous()
            elif save_X_or_KV == 1 and save_local_or_global == 1:
                Kj, Vj = KV_chunks[step]
                Kj, Vj = Kj.contiguous(), Vj.contiguous()
            elif save_X_or_KV == 0 and save_local_or_global == 0: 
                # some communication would be required here
                Xj = X_chunks[step]
                QKVj = (
                    (Xj @ W_qkv[d * h :, :].mT)
                    .reshape(B, Nlocal, 2, H, d)
                    .permute(2, 0, 1, 3, 4)
                )
            elif save_X_or_KV == 1 and save_local_or_global == 1: 
                Xj = X_chunks[step]
                QKVj = (
                    (Xj @ W_qkv[d * h :, :].mT)
                    .reshape(B, Nlocal, 2, H, d)
                    .permute(2, 0, 1, 3, 4)
                )
            
            Kj, Vj = KV_chunks[step]
            Kj = Kj.contiguous()
            Vj = Vj.contiguous()
            # L for this rank was saved as (B*H, Nlocal) — need full L not per-step
            # Use saved global L
            dQ_j, dKj, dVj = (
                torch.ops.aten._scaled_dot_product_flash_attention_backward(
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
            )
            dQ += dQ_j

            if home_rank == rank:
                dK += dKj
                dV += dVj
            
            # Continue to send stuff backwards.
            if save_local_or_global == 1:
                if save_X_or_KV == 0:
                    recv_buf = torch.zeros_like(Xj)
                    # Do I need to send Xj backward?
                    recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty)
                if save_X_or_KV == 1:
                    recv_buf = torch.zeros_like(KV_chunks[0]) # already have a local KV. Syntax may vary wrt. dimension.
                    # Do I need to send Xj backward?
                    recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty)


        return dQ, dK, dV, None, None, None, None, None, None, None, None