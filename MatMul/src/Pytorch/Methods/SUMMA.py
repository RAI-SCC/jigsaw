import torch
import torch.distributed as dist

import MatMul.src.Pytorch.Utility.ProcManager as ProcManager

# ProcessMesh pytorch
def ComputeGroups(grid_shape):
    rank = dist.get_rank()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)

    r, c = coords

    # Setup communication groups

    for i in range(num_r):
        row_ranks = [ProcManager.GetRank((i, j), grid_shape) for j in range(num_c)]
        row_group = dist.new_group(row_ranks)
        if i == r:
            comm_row = row_group
            
    for j in range(num_c):
        col_ranks = [ProcManager.GetRank((i, j), grid_shape) for i in range(num_r)]
        col_group = dist.new_group(col_ranks)
        if j == c:
            comm_col = col_group

    return comm_row, comm_col


def X_W(X, W, grid_shape, device, comm):
    rank = dist.get_rank()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = comm

    sub_n = X.shape[0]
    sub_k = X.shape[1]
    sub_m = W.shape[1]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)

    cx = torch.zeros((sub_n, sub_k), dtype=X.dtype, device=device)
    cw = torch.zeros((sub_k, sub_m), dtype=W.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(X)
        
        if r == k_idx:
            cw.copy_(W)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx @ cw
        
    return result_local

def XT_W(X, W, grid_shape, device, comm):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = comm

    # All_to_all for transposition
    # Compute target
    t_rank = ProcManager.GetRank((c, r), grid_shape)

    # Each rank creates a list of 4 tensors (assuming world_size=4)
    input_list = [torch.empty(0, device=device) for _ in range(world_size)]

    input_list[t_rank] = X

    # The output list must also be initialized
    output_list = [torch.empty(0, device=device) for _ in range(world_size)]
    # IMPORTANT: The rank receiving from Rank 1 must have an appropriately sized tensor here
    output_list[t_rank] = torch.empty_like(X) 

    dist.all_to_all(output_list, input_list, group=dist.group.WORLD)
    X = output_list[t_rank]

    sub_n = X.shape[1]
    sub_k = X.shape[0]
    sub_m = W.shape[1]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)

    cx = torch.zeros((sub_k, sub_n), dtype=X.dtype, device=device)
    cw = torch.zeros((sub_k, sub_m), dtype=W.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(X)
        
        if r == k_idx:
            cw.copy_(W)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx.T @ cw
        
    return result_local

def X_WT(X, W, grid_shape, device, comm):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = comm

    # All_to_all for transposition
    # Compute target
    t_rank = ProcManager.GetRank((c, r), grid_shape)

    # Each rank creates a list of 4 tensors (assuming world_size=4)
    input_list = [torch.empty(0, device=device) for _ in range(world_size)]

    input_list[t_rank] = W

    # The output list must also be initialized
    output_list = [torch.empty(0, device=device) for _ in range(world_size)]
    # IMPORTANT: The rank receiving from Rank 1 must have an appropriately sized tensor here
    output_list[t_rank] = torch.empty_like(W) 

    dist.all_to_all(output_list, input_list, group=dist.group.WORLD)
    W = output_list[t_rank]

    sub_n = X.shape[0]
    sub_k = X.shape[1]
    sub_m = W.shape[0]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)

    cx = torch.zeros((sub_n, sub_k), dtype=X.dtype, device=device)
    cw = torch.zeros((sub_m, sub_k), dtype=W.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(X)
        
        if r == k_idx:
            cw.copy_(W)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx @ cw.T
        
    return result_local