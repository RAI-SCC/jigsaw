import torch
import torch.distributed as dist

import MatMul.src.Pytorch.Utility.MatrixManager as MatrixManager
import MatMul.src.Pytorch.Utility.ProcManager as ProcManager

def X_W(X, W, grid_shape, device):
    rank = dist.get_rank()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    sub_n = X.shape[0]
    sub_m = W.shape[1]
    
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)
    
    # Pre-allocate static buffers to avoid CUDA malloc/free overhead
    # We need two buffers: one for the incoming X data and one for W
    X_buffer = torch.empty_like(X)
    W_buffer = torch.empty_like(W)

    # Identify Neighbors
    left_rank  = ProcManager.GetRank((r, (c - 1) % num_c), grid_shape)
    right_rank = ProcManager.GetRank((r, (c + 1) % num_c), grid_shape)
    up_rank    = ProcManager.GetRank(((r - 1) % num_r, c), grid_shape)
    down_rank  = ProcManager.GetRank(((r + 1) % num_r, c), grid_shape)

    def shift_inplace(send_tensor, recv_buffer, dest_rank, src_rank):
        """
        Uses batch_isend_irecv to safely swap data. 
        This prevents the 'NoneType' error and avoids deadlocks.
        """
        # Create a list of P2P operations
        send_op = dist.P2POp(dist.isend, send_tensor, dest_rank)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, src_rank)
        
        # batch_isend_irecv returns a list of handles
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        
        # Now we wait on the handles
        for req in reqs:
            if req is not None:
                req.wait()
                
        return recv_buffer, send_tensor

    # 3b. Initial Alignment
    # X: Shift left by 'r' positions
    for _ in range(r):
        X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
    
    # W: Shift up by 'c' positions
    for _ in range(c):
        W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    # 3c. Shift-and-Multiply Loop
    for i in range(num_c):
        # Perform compute (GEMM)
        # Using addmm or @ with +=
        result_local.addmm_(X, W)

        # Skip the final shift on the last iteration to save time
        if i < num_c - 1:
            X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
            W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    return result_local

def XT_W(X, W, grid_shape, device):
    rank = dist.get_rank()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    # IMPORTANT: Since X is "X^T positionally", X.shape[0] is the 
    # original number of columns. After X.t(), it matches W's rows.
    # result_local shape: (Rows of X^T, Cols of W)
    sub_n = X.shape[1] # This is rows of X^T
    sub_m = W.shape[1] # This is cols of W
    
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)
    
    # Pre-allocate static buffers
    # X_buffer matches X (the non-transposed block shape)
    X_buffer = torch.empty_like(X)
    W_buffer = torch.empty_like(W)

    # Identify Neighbors (Standard Cannon grid)
    left_rank  = ProcManager.GetRank((r, (c - 1) % num_c), grid_shape)
    right_rank = ProcManager.GetRank((r, (c + 1) % num_c), grid_shape)
    up_rank    = ProcManager.GetRank(((r - 1) % num_r, c), grid_shape)
    down_rank  = ProcManager.GetRank(((r + 1) % num_r, c), grid_shape)

    def shift_inplace(send_tensor, recv_buffer, dest_rank, src_rank):
        """
        Uses batch_isend_irecv to safely swap data. 
        This prevents the 'NoneType' error and avoids deadlocks.
        """
        # Create a list of P2P operations
        send_op = dist.P2POp(dist.isend, send_tensor, dest_rank)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, src_rank)
        
        # batch_isend_irecv returns a list of handles
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        
        # Now we wait on the handles
        for req in reqs:
            if req is not None:
                req.wait()
                
        return recv_buffer, send_tensor

    # --- 1. Initial Alignment ---
    # Standard Cannon: X shifts left by r, W shifts up by c
    # This works because X is already in its "transposed" row/col position
    for _ in range(r):
        X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
    
    for _ in range(c):
        W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    # --- 2. Shift-and-Multiply Loop ---
    for i in range(num_c):
        # Local GEMM: We transpose X locally here to align the math
        # X.t() is a view, so this is efficient.
        result_local.addmm_(X.t(), W)

        if i < num_c - 1:
            # Shift X left, W up
            X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
            # Use W_buffer to shift W up
            W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    return result_local

def X_WT(X, W, grid_shape, device):
    rank = dist.get_rank()
    num_r, num_c = grid_shape
    coords = ProcManager.GetCoords(rank, grid_shape)
    r, c = coords

    # If W is "positionally transposed" to represent W^T:
    # W.shape[1] is actually the original number of rows of W.
    # W.t().shape[1] will be the original number of columns of W.
    sub_n = X.shape[0]     # Rows of X
    sub_m = W.shape[0]     # Columns of W^T (which were Rows of W)
    
    result_local = torch.zeros((sub_n, sub_m), dtype=torch.float32, device=device)
    
    # Pre-allocate static buffers
    X_buffer = torch.empty_like(X)
    W_buffer = torch.empty_like(W)

    # Identify Neighbors (Standard Cannon grid)
    left_rank  = ProcManager.GetRank((r, (c - 1) % num_c), grid_shape)
    right_rank = ProcManager.GetRank((r, (c + 1) % num_c), grid_shape)
    up_rank    = ProcManager.GetRank(((r - 1) % num_r, c), grid_shape)
    down_rank  = ProcManager.GetRank(((r + 1) % num_r, c), grid_shape)

    def shift_inplace(send_tensor, recv_buffer, dest_rank, src_rank):
        """
        Uses batch_isend_irecv to safely swap data. 
        This prevents the 'NoneType' error and avoids deadlocks.
        """
        # Create a list of P2P operations
        send_op = dist.P2POp(dist.isend, send_tensor, dest_rank)
        recv_op = dist.P2POp(dist.irecv, recv_buffer, src_rank)
        
        # batch_isend_irecv returns a list of handles
        reqs = dist.batch_isend_irecv([send_op, recv_op])
        
        # Now we wait on the handles
        for req in reqs:
            if req is not None:
                req.wait()
                
        return recv_buffer, send_tensor

    # --- 1. Initial Alignment ---
    # X: Shift left by r positions
    for _ in range(r):
        X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
    
    # W: Shift up by c positions (W here is the positionally transposed W^T)
    for _ in range(c):
        W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    # --- 2. Shift-and-Multiply Loop ---
    for i in range(num_c):
        # Local GEMM: X @ W^T
        # We use W.t() because the blocks are in the right place, 
        # but the data inside them needs transposing.
        result_local.addmm_(X, W.t())

        if i < num_c - 1:
            X, X_buffer = shift_inplace(X, X_buffer, left_rank, right_rank)
            W, W_buffer = shift_inplace(W, W_buffer, up_rank, down_rank)

    return result_local