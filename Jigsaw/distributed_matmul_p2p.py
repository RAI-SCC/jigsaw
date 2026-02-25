import torch
import torch.distributed as dist

from .distributed_matmul import _get_preallocated_buffer

def xw_transpose_twoway(
    xij: torch.tensor,
    wij: torch.tensor,
    global_rank: int,
    device: torch.device,
    model_group: dist.ProcessGroup,
) -> torch.tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.

    Parameters
    ----------
    xij : torch.tensor
        Local input matrix.
    wij : torch.tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    local_rank : int
        Local local_rank.
    model_group : dist.ProcessGroup, optional
        Process group for an individual model-parallel instance.

    Returns
    -------
    cij : torch.tensor
        Local result.
    """
    mod = global_rank % 2

    w0, w1 = torch.split(wij, wij.shape[-2] // 2, dim=-2)
    # print(_prealloc_buffers.keys())
    buffer_xw = _get_preallocated_buffer(
        (*xij.shape[:-1], w0.shape[-2]), torch.float32, global_rank, device
    )
    
    match mod:
        case 0:
            xw = torch.matmul(xij, w1.mT)
            dist.isend(xw, dst=global_rank+1, group=model_group)
            recv = dist.irecv(buffer_xw, src=global_rank+1, group=model_group)
        case 1:
            xw = torch.matmul(xij, w0.mT)
            dist.isend(xw, dst=global_rank-1, group=model_group)
            recv = dist.irecv(buffer_xw, src=global_rank-1, group=model_group)
            
    match mod:
        case 0:
            cij = torch.matmul(xij, w0.mT)
            recv.wait()
            cij = cij + buffer_xw
        case 1:
            cij = torch.matmul(xij, w1.mT)
            recv.wait()
            cij = cij + buffer_xw
    
    return cij

def XTW_twoway_(Xij, Wij, rank, device, model_group):
    """
    Batched matrix-matrix (XTW) multiplication distributed across two processes.

    Parameters
    ----------
    Xij : torch.Tensor
        Local input matrix.
    Wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        The device on which to perform the operations.
    rank : int
        Global rank.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.
    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """    
    mod = rank % 2

    bufferX = _get_preallocated_buffer(Xij.shape, torch.float32, rank, device)
    
    if mod == 0: 
        send_B_1 = torch.distributed.isend(Xij, dst=rank+1, group=model_group)
        recv_B_1 = torch.distributed.irecv(bufferX, src=rank+1, group=model_group)

        Cij_top = torch.matmul(Xij.mT, Wij)
        send_B_1.wait()
        recv_B_1.wait()
        
    elif mod == 1:
        Cij = torch.matmul(Xij, w_recv_list[5].mT) + torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 2:
        Cij = torch.matmul(Xij, Wij.mT) + torch.matmul(Xij, w_recv_list[0].mT)
    elif mod == 3:
        Cij = torch.matmul(Xij, w_recv_list[5].mT) + torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 4:
        Cij = torch.matmul(Xij, w_recv_list[0].mT) + torch.matmul(Xij, w_recv_list[2].mT)
    elif mod == 5:
        Cij = torch.matmul(Xij, Wij.mT) + torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 6:
        Cij = torch.matmul(Xij, w_recv_list[0].mT) + torch.matmul(Xij, w_recv_list[2].mT)
    elif mod == 7:
        Cij = torch.matmul(Xij, w_recv_list[5].mT) + torch.matmul(Xij, Wij.mT)
        
    handle_1.wait()
    
    if mod == 0:
        Cij_bottom = torch.matmul(bufferX.mT, Wij)
    else:
        Cij_top = torch.matmul(bufferX.mT, Wij)

    Cij = torch.concat([Cij_top, Cij_bottom], dim=-2)
    return Cij


def matrix_transpose_twoway_(X, rank, device, model_group):
    """
    Batched transpose (XT) operation distributed across two processes.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    rank : int
        Global rank.
    device : torch.device
        Device to perform computation on.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.
    
    Returns
    -------
    XT: torch.Tensor
        Local transposed tensor.
    """

    mod = rank % 2

    X0, X1 = torch.split(X, X.shape[-2]//2, dim=-2)
    bufferX = _get_preallocated_buffer(X1.shape, torch.float32, rank, device) 
    
    if mod == 0: 
        send_B_1 = torch.distributed.isend(X1, dst=rank+1, group=model_group)
        recv_B_1 = torch.distributed.irecv(bufferX, src=rank+1, group=model_group)
        X0 = X0.mT    
        if len(X0.shape) == 3:
            XT = torch.empty(X0.shape[0], X0.shape[1]*2, X0.shape[2], device=device)
            XT[:, :X0.shape[1]] = X0
        else:
            XT = torch.empty(X0.shape[0]*2, X0.shape[1], device=device)
            XT[:X0.shape[1]] = X0

        send_B_1.wait()
        recv_B_1.wait()
        
    elif mod == 1:
        recv_transpose = torch.distributed.irecv(bufferX, src=rank-1, group=model_group)
        send_transpose = torch.distributed.isend(X0, dst=rank-1, group=model_group)
        X1 = X1.mT
        if len(X0.shape) == 3:
            XT = torch.empty(X1.shape[0], X1.shape[1]*2, X1.shape[2], device=device)
            XT[:,X1.shape[1]:] = X1
        else:
            XT = torch.empty(X1.shape[1]*2, X1.shape[2], device=device)
            XT[X1.shape[1]:] = X1

        send_transpose.wait()
        recv_transpose.wait()
        
        
    if mod == 0:
        if len(X0.shape) == 3:
            XT[:, X0.shape[1]:, :] = bufferX.mT
        else:
            XT[X0.shape[1]:, :] = bufferX.mT
    elif mod == 1:
        if len(X1.shape) == 3:
            XT[:, :X1.shape[1], :] = bufferX.mT
        else:
            XT[:X1.shape[1], :] = bufferX.mT
    return XT

def XW_twoway_(Xij, Wij, rank, device, model_group):
    """
    Batched matrix-matrix (XW) multiplication distributed across two processes.

    Parameters
    ----------
    Xij : torch.Tensor
        Local input matrix.
    Wij : torch.Tensor
        Local weight matrix.
    device : torch.device
        Device to perform computation on.
    rank : int
        Global rank.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.
        
    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """
    
    mod = rank % 2
    W0, W1 = torch.split(Wij, Wij.shape[-2]//2, dim=-2)
    bufferX = _get_preallocated_buffer(Xij.shape, torch.float32, rank, device) 
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_W_1 = torch.distributed.isend(Xij, dst=rank+1, group=None)
        recv_W_1 = torch.distributed.irecv(bufferX, src=rank+1, group=None)

        Cij = torch.matmul(Xij, W0)
        send_W_1.wait()
        recv_W_1.wait()
    elif mod == 1:
        recv_W_0 = torch.distributed.irecv(bufferX, src=rank-1, group=None)
        send_W_0 = torch.distributed.isend(Xij, dst=rank-1, group=None)
        
        Cij = torch.matmul(Xij, W1)
        send_W_0.wait()
        recv_W_0.wait()
    
    
    if mod == 0:
        Cij = Cij + torch.matmul(bufferX, W1)
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferX, W0)
    return Cij 

def XW_eightway_(Xij, Wij, rank,  device, model_group):
    
    """
    Batched matrix-matrix (XW) multiplication distributed across eight processes.
    Parameters
    ----------
    Xij : torch.Tensor
        Local input matrix.
    Wij : torch.Tensor
        Local weight matrix.
    rank : int
        Global rank.
    device : torch.device
        Device to perform computation on.
    model_group : torch.distributed.ProcessGroup, optional
        Process group for an individual model-parallel instance.
    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """
    
    mod = rank % 8
    
    empty = torch.empty(0, device=device, dtype=Xij.dtype)
    
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1),
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3)
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1),
            empty,
            empty,
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty, 
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3)
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2),
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3)
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0)]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2),
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty,
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, Xij, empty]
        x_recv_list = [empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty,
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    #X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # EXAMPLE SPLITTING CODE

    handle_w.wait()
    
    if mod == 0:
        ws_stacked = torch.concatenate([Wij, w_recv_list[2]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[2])
    elif mod == 1:
        ws_stacked = torch.concatenate([w_recv_list[5], w_recv_list[7]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, w_recv_list[7])
    elif mod == 2:
        ws_stacked = torch.concatenate([w_recv_list[0], Wij], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, Wij)
    elif mod == 3:
        ws_stacked = torch.concatenate([w_recv_list[5], w_recv_list[7]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, w_recv_list[7])
    elif mod == 4:
        ws_stacked = torch.concatenate([w_recv_list[0], w_recv_list[2]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, w_recv_list[2])
    elif mod == 5:
        ws_stacked = torch.concatenate([Wij, w_recv_list[7]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[7])
    elif mod == 6:
        ws_stacked = torch.concatenate([w_recv_list[0], w_recv_list[2]], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[0]) + torch.matmul(X1, w_recv_list[2])
    elif mod == 7:
        ws_stacked = torch.concatenate([Wij, Wij], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, Wij)  
    
    Cij = torch.matmul(Xij, ws_stacked)
    handle_x.wait()
    
    if mod == 0:
        X0, X1 = torch.split(x_recv_list[1], x_recv_list[1].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[4]) + torch.matmul(X1, w_recv_list[6])
    elif mod == 1:
        X0, X1 = torch.split(x_recv_list[0], x_recv_list[0].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[3])
    elif mod == 2:
        X0, X1 = torch.split(x_recv_list[3], x_recv_list[3].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[4]) + torch.matmul(X1, w_recv_list[6])
    elif mod == 3:
        X0, X1 = torch.split(x_recv_list[2], x_recv_list[2].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[1]) + torch.matmul(X1, Wij)
    elif mod == 4:
        X0, X1 = torch.split(x_recv_list[5], x_recv_list[5].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, Wij) + torch.matmul(X1, w_recv_list[6])
    elif mod == 5:
        X0, X1 = torch.split(x_recv_list[4], x_recv_list[4].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[1]) + torch.matmul(X1, w_recv_list[3])
    elif mod == 6:
        X0, X1 = torch.split(x_recv_list[7], x_recv_list[7].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[4]) + torch.matmul(X1, Wij)
    elif mod == 7:
        X0, X1 = torch.split(x_recv_list[6], x_recv_list[6].shape[-1]//2, dim=-1) # split upon last dimension
        Cij = Cij + torch.matmul(X0, w_recv_list[1]) + torch.matmul(X1, w_recv_list[3])

    return Cij
