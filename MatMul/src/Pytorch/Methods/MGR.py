import torch
import torch.distributed as dist

import MatMul.src.Pytorch.Utility.MatrixManager as MatrixManager

def X_W(X, W, device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    sub_n = X.shape[0]
    sub_m = W.shape[1]
    m = sub_m * world_size 

    # n=100, k=150, m=60 -> world_size=4
    # sub_n = 25 (righe di X)
    # sub_m = 15 (colonne di W)
    # sub_n = n // world_size
    # sub_m = m // world_size
    
    # # 1. CARICAMENTO INIZIALE
    # # X è (n x k). Carichiamo RIGHE: lines_per_block=sub_n, total_lines=n, other_dim=k
    # X = MatrixManager.ReadLine(x_path, rank, sub_n, n, k, isRows=True)

    # # W è (k x m). Carichiamo COLONNE: lines_per_block=sub_m, total_lines=m, other_dim=k
    # # Nota: W cambierà ad ogni step del ring
    # W = MatrixManager.ReadLine(w_path, rank, sub_m, m, k, isRows=False)
    
    # Il risultato locale conterrà le sub_n righe per tutte le m colonne
    result_local = torch.zeros((sub_n, m), dtype=torch.float32, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # 3. CICLO RING
    W_next = torch.empty_like(W)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        result_local[:, col_start:col_end] = X @ W

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, W, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            W.copy_(W_next)
        
    return result_local

def XT_W(X, W, device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    sub_n = X.shape[1]
    sub_m = W.shape[1]
    m = sub_m * world_size 
    
    # Il risultato locale conterrà le sub_n righe per tutte le m colonne
    result_local = torch.zeros((sub_n // world_size, m), dtype=torch.float32, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # all_to_all transposition of X (inefficient reallocation of memory, but required 
    #    because of the choice of having the result as (n/p * m) (row division))
    # send_buf = list(X.view(X.shape[0], world_size, sub_n // world_size).permute(1, 0, 2).contiguous().unbind(0))
    # recv_buf = [torch.zeros_like(send_buf[0]) for _ in range(world_size)]

    # dist.all_to_all(recv_buf, send_buf, dist.group.WORLD)

    # X = torch.cat(recv_buf).T

    X = X.view(X.shape[0], world_size, sub_n // world_size).permute(1, 0, 2).contiguous()
    recv_buf = torch.empty_like(X)

    dist.all_to_all_single(recv_buf, X, group=dist.group.WORLD)

    X = recv_buf.permute(2, 0, 1).reshape(X.shape[2], -1)

    # if rank == 0:
    #     print(f"X shape: {X.shape}\nW Shape: {W.shape}\nRes shape: {result_local[:, 0:sub_m].shape}")

    # 3. CICLO RING
    W_next = torch.empty_like(W)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        res = X @ W
        # print(f"Res {step}: {res.shape}")
        result_local[:, col_start:col_end] = res

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, W, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            W.copy_(W_next)
        
    return result_local

def X_WT(X, W, device):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    sub_n = X.shape[0]
    sub_m = W.shape[0] // world_size
    m = sub_m * world_size
    
    # Il risultato locale conterrà le sub_n righe per tutte le m colonne
    result_local = torch.zeros((sub_n, m), dtype=torch.float32, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # all_to_all transposition of X (inefficient reallocation of memory, but required)
    # send_buf = list(W.view(world_size, sub_m, W.shape[1]).unbind(0))
    # recv_buf = [torch.zeros_like(send_buf[0]) for _ in range(world_size)]

    # dist.all_to_all(recv_buf, send_buf, dist.group.WORLD)

    # W = torch.cat(recv_buf, dim=1)

    W = W.view(world_size, sub_m, W.shape[1])
    
    # 2. Unica allocazione per il buffer di ricezione
    recv_buf = torch.empty_like(W)

    # 3. Comunicazione collettiva a puntatore singolo
    dist.all_to_all_single(recv_buf, W, group=dist.group.WORLD)

    # 4. Reshape a costo zero (sostituisce il vecchio torch.cat)
    W = recv_buf.permute(1, 0, 2).reshape(sub_m, -1)

    # if rank == 0:
    #     print(f"X shape: {X.shape}\nW Shape: {W.shape}")

    # 3. CICLO RING
    W_next = torch.empty_like(W)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        result_local[:, col_start:col_end] = X @ W.T

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, W, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            W.copy_(W_next)
        
    return result_local