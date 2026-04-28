import torch
import torch.distributed as dist
import numpy as np

import os

def Read(path, geometric_rank, sub_rows, sub_cols, total_rows, total_cols, dtype=np.float32):
    """
    path: Path to the .dat file
    geometric_rank: (r, c) - processor grid coordinates
    sub_rows, sub_cols: Dimensions of the local submatrix
    total_rows, total_cols: Number of submatrices in the grid
    """
    item_size = np.dtype(dtype).itemsize
    r, c = geometric_rank
    
    # 1. Calculate how many elements exist before our submatrix row
    # Each 'row' of submatrices is (sub_rows * (total_cols * sub_cols)) elements
    elements_per_sub_row = sub_rows * (total_cols * sub_cols)
    
    # 2. Calculate the offset within the row
    # Each 'col' of submatrices is (sub_cols) elements
    offset_in_elements = (r * elements_per_sub_row) + (c * sub_cols)
    
    # 3. Use memory map to access only the required slice
    # We need to map the whole file but only read our piece
    full_matrix_shape = (total_rows * sub_rows, total_cols * sub_cols)
    
    # Memory map the file
    mm = np.memmap(path, dtype=dtype, mode='r', shape=full_matrix_shape)
    
    # Extract the local submatrix
    # [start_row:end_row, start_col:end_col]
    local_data = mm[r*sub_rows : (r+1)*sub_rows, c*sub_cols : (c+1)*sub_cols]
    
    # Convert to torch tensor
    # .copy() is required because memmap is not natively a torch tensor
    return torch.from_numpy(np.array(local_data)).clone()

def Write(submatrix: torch.Tensor, geometric_rank, total_rows, total_cols, path, dtype=np.float32):
    r, c = geometric_rank
    sub_rows, sub_cols = submatrix.shape
    
    # 1. Calculate file size in bytes
    item_size = np.dtype(dtype).itemsize
    total_elements = (total_rows * sub_rows) * (total_cols * sub_cols)
    file_size_bytes = total_elements * item_size
    
    # 2. Pre-allocate the file using ftruncate (Only rank 0 does this)
    if dist.get_rank() == 0:
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.truncate(file_size_bytes)
    
    # Barrier to ensure the file exists and is sized correctly before others open it
    dist.barrier()
    
    # 3. Map the file
    full_shape = (total_rows * sub_rows, total_cols * sub_cols)
    mm = np.memmap(path, dtype=dtype, mode='r+', shape=full_shape)
    
    # 4. Write the submatrix
    start_row, end_row = r * sub_rows, (r + 1) * sub_rows
    start_col, end_col = c * sub_cols, (c + 1) * sub_cols
    
    mm[start_row:end_row, start_col:end_col] = submatrix.detach().cpu().numpy()
    
    # 5. Flush and barrier
    mm.flush()
    dist.barrier()

def Compare(A, B, tol=1e-6):
    """
    A, B: Local torch.Tensors (submatrices)
    tol: Tolerance for floating point comparison
    Returns: Boolean (True if globally equal)
    """
    # 1. Local comparison: Check if elements are close enough
    # We use torch.allclose for robustness against floating point errors
    local_equal = torch.allclose(A, B, atol=tol)
    
    # 2. Prepare result for reduction
    # Create a tensor representing local result (1 for True, 0 for False)
    result_tensor = torch.tensor(1 if local_equal else 0, device=A.device)
    
    # 3. Perform Global Reduction
    # all_reduce with SUM will result in (world_size) if all are equal
    dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
    
    # 4. Final check: Does the sum equal the number of processes?
    is_globally_equal = (result_tensor.item() == dist.get_world_size())
    
    return is_globally_equal

def ReadLine(path, rank, lines_per_block, total_lines, other_dim, isRows=True, dtype=np.float32):
    """
    Legge una striscia di matrice dal file .dat.
    
    Se isRows=True:  Legge una striscia di righe (es. W_local). 
                     Shape nel file: (total_lines, other_dim)
    Se isRows=False: Legge una striscia di colonne (es. X_current). 
                     Shape nel file: (other_dim, total_lines)
    """
    # 1. Definiamo la shape REALE del file e la slice corretta
    if isRows:
        full_shape = (total_lines, other_dim)
        slice_idx = (slice(rank * lines_per_block, (rank + 1) * lines_per_block), slice(None))
    else:
        full_shape = (other_dim, total_lines)
        slice_idx = (slice(None), slice(rank * lines_per_block, (rank + 1) * lines_per_block))

    # 2. Memory map in modalità sola lettura
    mm = np.memmap(path, dtype=dtype, mode='r', shape=full_shape)
    
    try:
        # 3. Estrazione e copia in RAM
        # .copy() è fondamentale per slegare il tensore dal file mappato
        # np.array() assicura la conversione corretta prima di torch
        local_data = mm[slice_idx]
        local_tensor = torch.from_numpy(np.array(local_data)).clone()
    finally:
        # Forza il rilascio dell'handle del file (critico su Windows)
        del mm

    return local_tensor

def WriteLine(submatrix, path, rank, lines_per_block, total_lines, other_dim, isRows=True, dtype=np.float32):
    """
    Scrive una striscia di matrice nel file .dat.
    """
    item_size = np.dtype(dtype).itemsize
    total_elements = total_lines * other_dim
    file_size_bytes = total_elements * item_size
    
    if isRows:
        full_shape = (total_lines, other_dim)
        slice_idx = (slice(rank * lines_per_block, (rank + 1) * lines_per_block), slice(None))
    else:
        full_shape = (other_dim, total_lines)
        slice_idx = (slice(None), slice(rank * lines_per_block, (rank + 1) * lines_per_block))

    # Il Rank 0 prepara il file se non esiste
    if dist.get_rank() == 0:
        if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != '':
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.truncate(file_size_bytes)
    
    dist.barrier()
    
    # Mappatura in lettura/scrittura
    mm = np.memmap(path, dtype=dtype, mode='r+', shape=full_shape)
    
    try:
        mm[slice_idx] = submatrix.detach().cpu().numpy()
        mm.flush()
    finally:
        del mm
        
    dist.barrier()


def ReadT(path, geometric_rank, sub_rows, sub_cols, total_rows, total_cols, dtype=np.float32):
    """
    path: Path to the .dat file
    geometric_rank: (r, c) - processor grid coordinates
    sub_rows, sub_cols: Dimensions of the local submatrix
    total_rows, total_cols: Number of submatrices in the grid
    """
    item_size = np.dtype(dtype).itemsize
    c, r = geometric_rank
    
    # 1. Calculate how many elements exist before our submatrix row
    # Each 'row' of submatrices is (sub_rows * (total_cols * sub_cols)) elements
    elements_per_sub_row = sub_rows * (total_cols * sub_cols)
    
    # 2. Calculate the offset within the row
    # Each 'col' of submatrices is (sub_cols) elements
    offset_in_elements = (r * elements_per_sub_row) + (c * sub_cols)
    
    # 3. Use memory map to access only the required slice
    # We need to map the whole file but only read our piece
    full_matrix_shape = (total_rows * sub_rows, total_cols * sub_cols)
    
    # Memory map the file
    mm = np.memmap(path, dtype=dtype, mode='r', shape=full_matrix_shape)
    
    # Extract the local submatrix
    # [start_row:end_row, start_col:end_col]
    local_data = mm[r*sub_rows : (r+1)*sub_rows, c*sub_cols : (c+1)*sub_cols]
    
    # Convert to torch tensor
    # .copy() is required because memmap is not natively a torch tensor
    return torch.from_numpy(np.array(local_data)).clone()