import torch
import torch.distributed as dist

def x_w_mgr(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = local_rank
    world_size = dist.get_world_size()
    
    sub_n = xij.shape[0]
    sub_m = wij.shape[1]
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
    result_local = torch.zeros((sub_n, m), dtype=xij.dtype, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # 3. CICLO RING
    W_next = torch.empty_like(wij)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        result_local[:, col_start:col_end] = xij @ wij

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, wij, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            wij.copy_(W_next)
        
    return result_local

def x_transpose_w_mgr(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = local_rank
    world_size = dist.get_world_size()
    
    sub_n = xij.shape[1]
    sub_m = wij.shape[1]
    m = sub_m * world_size 
    
    # Il risultato locale conterrà le sub_n righe per tutte le m colonne
    result_local = torch.zeros((sub_n // world_size, m), dtype=xij.dtype, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # all_to_all transposition of X (inefficient reallocation of memory, but required 
    #    because of the choice of having the result as (n/p * m) (row division))
    # send_buf = list(X.view(X.shape[0], world_size, sub_n // world_size).permute(1, 0, 2).contiguous().unbind(0))
    # recv_buf = [torch.zeros_like(send_buf[0]) for _ in range(world_size)]

    # dist.all_to_all(recv_buf, send_buf, dist.group.WORLD)

    # X = torch.cat(recv_buf).T

    xij = xij.view(xij.shape[0], world_size, sub_n // world_size).permute(1, 0, 2).contiguous()
    recv_buf = torch.empty_like(xij)

    dist.all_to_all_single(recv_buf, xij, group=dist.group.WORLD)

    xij = recv_buf.permute(2, 0, 1).reshape(xij.shape[2], -1)

    # if rank == 0:
    #     print(f"X shape: {X.shape}\nW Shape: {W.shape}\nRes shape: {result_local[:, 0:sub_m].shape}")

    # 3. CICLO RING
    W_next = torch.empty_like(wij)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        res = xij @ wij
        # print(f"Res {step}: {res.shape}")
        result_local[:, col_start:col_end] = res

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, wij, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            wij.copy_(W_next)
        
    return result_local

def x_w_transpose_mgr(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = local_rank
    world_size = dist.get_world_size()
    
    sub_n = xij.shape[0]
    sub_m = wij.shape[0] // world_size
    m = sub_m * world_size
    
    # Il risultato locale conterrà le sub_n righe per tutte le m colonne
    result_local = torch.zeros((sub_n, m), dtype=xij.dtype, device=device)

    # 2. SETUP DELL'ANELLO
    left_neighbor = (rank - 1) % world_size
    right_neighbor = (rank + 1) % world_size

    # all_to_all transposition of X (inefficient reallocation of memory, but required)
    # send_buf = list(W.view(world_size, sub_m, W.shape[1]).unbind(0))
    # recv_buf = [torch.zeros_like(send_buf[0]) for _ in range(world_size)]

    # dist.all_to_all(recv_buf, send_buf, dist.group.WORLD)

    # W = torch.cat(recv_buf, dim=1)

    wij = wij.view(world_size, sub_m, wij.shape[1])
    
    # 2. Unica allocazione per il buffer di ricezione
    recv_buf = torch.empty_like(wij)

    # 3. Comunicazione collettiva a puntatore singolo
    dist.all_to_all_single(recv_buf, wij, group=dist.group.WORLD)

    # 4. Reshape a costo zero (sostituisce il vecchio torch.cat)
    wij = recv_buf.permute(1, 0, 2).reshape(sub_m, -1)

    # if rank == 0:
    #     print(f"X shape: {X.shape}\nW Shape: {W.shape}")

    # 3. CICLO RING
    W_next = torch.empty_like(wij)
    for step in range(world_size):
        # Calcolo dell'indice del blocco di colonne di W che abbiamo attualmente
        # Step 0: W_rank, Step 1: W_{rank-1}, etc.
        current_w_idx = (rank - step) % world_size
        
        # Posizionamento nel buffer del risultato
        col_start = current_w_idx * sub_m
        col_end = col_start + sub_m
        
        # Moltiplicazione: (sub_n x k) @ (k x sub_m) -> (sub_n x sub_m)
        # result_local è (25, 60), il prodotto è (25, 15)
        result_local[:, col_start:col_end] = xij @ wij.T

        # TRASMISSIONE DI W (Anello)
        if step < world_size - 1:
            # Use batch operations to match Cannon's method and prevent NoneType errors
            send_op = dist.P2POp(dist.isend, wij, right_neighbor)
            recv_op = dist.P2POp(dist.irecv, W_next, left_neighbor)
            
            reqs = dist.batch_isend_irecv([send_op, recv_op])
            
            for req in reqs:
                if req is not None:
                    req.wait()
            
            wij.copy_(W_next)
        
    return result_local