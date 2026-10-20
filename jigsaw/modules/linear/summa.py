import torch
import torch.distributed as dist

import jigsaw.modules.linear.ProcManager as ProcManager

def compute_groups(grid_shape):
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

def x_w_summa(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = local_rank

    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid_shape = (grid_size, grid_size)

    num_r, num_c = grid_shape

    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = model_group

    sub_n = xij.shape[0]
    sub_k = xij.shape[1]
    sub_m = wij.shape[1]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=xij.dtype, device=device)

    cx = torch.zeros((sub_n, sub_k), dtype=xij.dtype, device=device)
    cw = torch.zeros((sub_k, sub_m), dtype=xij.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(xij)
        
        if r == k_idx:
            cw.copy_(wij)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx @ cw
        
    return result_local

def x_transpose_w_summa(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid_shape = (grid_size, grid_size)


    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = model_group

    # All_to_all for transposition
    # Compute target
    t_rank = ProcManager.GetRank((c, r), grid_shape)

    # Each rank creates a list of 4 tensors (assuming world_size=4)
    input_list = [torch.empty(0, device=device) for _ in range(world_size)]

    input_list[t_rank] = xij

    # The output list must also be initialized
    output_list = [torch.empty(0, device=device) for _ in range(world_size)]
    # IMPORTANT: The rank receiving from Rank 1 must have an appropriately sized tensor here
    output_list[t_rank] = torch.empty_like(xij) 

    dist.all_to_all(output_list, input_list, group=dist.group.WORLD)
    xij = output_list[t_rank]

    sub_n = xij.shape[1]
    sub_k = xij.shape[0]
    sub_m = wij.shape[1]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=xij.dtype, device=device)

    cx = torch.zeros((sub_k, sub_n), dtype=xij.dtype, device=device)
    cw = torch.zeros((sub_k, sub_m), dtype=wij.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(xij)
        
        if r == k_idx:
            cw.copy_(wij)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx.T @ cw
        
    return result_local

def x_w_transpose_summa(
    xij: torch.Tensor,
    wij: torch.Tensor,
    local_rank: int,
    device: torch.device,
    model_group
) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid_shape = (grid_size, grid_size)

    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    comm_row, comm_col = model_group

    # All_to_all for transposition
    # Compute target
    t_rank = ProcManager.GetRank((c, r), grid_shape)

    # Each rank creates a list of 4 tensors (assuming world_size=4)
    input_list = [torch.empty(0, device=device) for _ in range(world_size)]

    input_list[t_rank] = wij

    # The output list must also be initialized
    output_list = [torch.empty(0, device=device) for _ in range(world_size)]
    # IMPORTANT: The rank receiving from Rank 1 must have an appropriately sized tensor here
    output_list[t_rank] = torch.empty_like(wij) 

    dist.all_to_all(output_list, input_list, group=dist.group.WORLD)
    wij = output_list[t_rank]

    sub_n = xij.shape[0]
    sub_k = xij.shape[1]
    sub_m = wij.shape[0]

    # Memory setup
    result_local = torch.zeros((sub_n, sub_m), dtype=xij.dtype, device=device)

    cx = torch.zeros((sub_n, sub_k), dtype=xij.dtype, device=device)
    cw = torch.zeros((sub_m, sub_k), dtype=wij.dtype, device=device)

    # SUMMA Loop
    for k_idx in range(num_c):
        # Calculate global ranks for the current panel owners
        # X is distributed such that rank (r, k_idx) owns the k_idx-th panel of row r
        src_row_rank = ProcManager.GetRank((r, k_idx), grid_shape)
        
        # W is distributed such that rank (k_idx, c) owns the k_idx-th panel of column c
        src_col_rank = ProcManager.GetRank((k_idx, c), grid_shape)

        # Copy local data to the broadcast buffer ONLY if this process is the owner
        if c == k_idx:
            cx.copy_(xij)
        
        if r == k_idx:
            cw.copy_(wij)

        # Broadcast panels within the row/column groups
        # All processes in comm_row receive the panel from src_row_rank
        dist.broadcast(cx, src=src_row_rank, group=comm_row)
        
        # All processes in comm_col receive the panel from src_col_rank
        dist.broadcast(cw, src=src_col_rank, group=comm_col)
        
        # Accumulate result
        result_local += cx @ cw.T
        
    return result_local