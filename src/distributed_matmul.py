import torch
import torch.distributed as dist
import torch.distributed


_prealloc_buffers = {}

def get_preallocated_buffer(shape, dtype, device, rank, index=0):
    """
    Retrieve or create a preallocated buffer.
    Parameters
    ----------
    shape : tuple
        The shape of the buffer to be created or retrieved.
    dtype : torch.dtype
        The data type of the buffer.
    device : torch.device
        Device to perform computation on.
    rank : int
        Global rank.
    index : int, optional
        An additional index to differentiate buffers (default is 0).
    Returns
    -------
    torch.Tensor
        Preallocated buffer.
    """

    key = (dtype, rank, index)
    if key not in _prealloc_buffers:
        _prealloc_buffers[key] = (None, 0)

    buf, buf_numel = _prealloc_buffers[key]

    needed = 1
    
    for dim in shape: 
        needed *= dim
    
    # Reallocate if buffer is None or too small
    if buf is None or needed > buf_numel:
        buf = torch.zeros(needed, dtype=dtype, device=device)
        buf_numel = needed
        _prealloc_buffers[key] = (buf, buf_numel)
        return buf.view(*shape)
    else:        
        out, _ = _prealloc_buffers[key]
        out = out[:needed].view(*shape)
        
        return out
    
## Two-way distributions
def matrix_transpose_twoway(X, rank, device, model_group=None):
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
    bufferX = get_preallocated_buffer(X1.shape, torch.float32, device, rank) 
    
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

def XWT_twoway(Xij, Wij, device, rank, model_group=None):
    """
    Batched matrix-matrix (XWT) multiplication distributed across two processes.

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
    bufferXW = get_preallocated_buffer((*Xij.shape[:-1], W0.shape[-2]), torch.float32, device, rank) 

    empty = torch.empty(0, device=device)
    

    if mod == 0:
        XW = torch.matmul(Xij, W1.mT)
        xw_send_list = [empty, XW]
        xw_recv_list = [empty, bufferXW]
    else:
        XW = torch.matmul(Xij, W0.mT)
        xw_send_list = [XW, empty]
        xw_recv_list = [bufferXW, empty]

    xw_handle = dist.all_to_all(xw_recv_list, xw_send_list, group=model_group, async_op=True)
        
    if mod == 0:
        Cij = torch.matmul(Xij, W0.mT)
        xw_handle.wait()
        Cij = Cij + xw_recv_list[1]
    else:
        Cij = torch.matmul(Xij, W1.mT)
        xw_handle.wait()
        Cij = Cij + xw_recv_list[0]
    
    return Cij



def XW_twoway(Xij, Wij, device, rank, model_group=None):
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
    bufferX = get_preallocated_buffer(Xij.shape, torch.float32, device, rank) 
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

def XTW_twoway(Xij, Wij, device, rank, model_group=None):
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

    bufferX = get_preallocated_buffer(Xij.shape, torch.float32, device, rank)
    
    if mod == 0: 
        send_B_1 = torch.distributed.isend(Xij, dst=rank+1, group=None)
        recv_B_1 = torch.distributed.irecv(bufferX, src=rank+1, group=None)

        Cij_top = torch.matmul(Xij.mT, Wij)
        send_B_1.wait()
        recv_B_1.wait()
        
    elif mod == 1:
        recv_B_0 = torch.distributed.irecv(bufferX, src=rank-1, group=None)
        send_B_0 = torch.distributed.isend(Xij, dst=rank-1, group=None)

        Cij_bottom = torch.matmul(Xij.mT, Wij)
        send_B_0.wait()
        recv_B_0.wait()

    if mod == 0:
        Cij_bottom = torch.matmul(bufferX.mT, Wij)
    else:
        Cij_top = torch.matmul(bufferX.mT, Wij)

    Cij = torch.concat([Cij_top, Cij_bottom], dim=-2)
    return Cij

## Four-way distributions
def matrix_transpose(X, rank, device, model_group=None):
    """
    Batched transpose (XT) operation distributed across four processes.
    Parameters
    ----------
    X : torch.Tensor
        Local input tensor to be transposed.
    rank : int
        Global rank.
    device : torch.device
        Device to perform computation on.
    Returns
    -------
    X : torch.Tensor
        The transposed tensor.
    """

    mod = rank % 4
    empty = torch.empty(0, device=device)
    if mod == 0 or mod == 3:
        x_send_list = [empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty]
    elif mod == 1:
        x_send_list = [empty, empty, X, empty]
        x_recv_list = [empty, empty, torch.empty_like(X), empty]
    else: # mod == 2
        x_send_list = [empty,  X, empty, empty]
        x_recv_list = [empty, torch.empty_like(X), empty, empty]

    dist.all_to_all(x_recv_list, x_send_list, group=model_group)
    
    if mod == 0 or mod == 3:
        return X.mT
    elif mod == 1:
        return x_recv_list[2].mT
    else:
        return x_recv_list[1].mT
    
def XWT_fourway(Xij, Wij, rank,  device, model_group=None):
    
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
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
    
    mod = rank % 4
    
    empty = torch.empty(0, device=device)

    if mod == 0:
        x_send_list = [empty, Xij, empty, empty]
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), empty, empty]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty]
        x_recv_list = [get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty]       
        w_recv_list = [empty, empty, get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1), get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2)]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)]
        w_send_list = [empty, Wij, empty, Wij]
        w_recv_list = [get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1), get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2), empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty, empty, get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), empty]

    handle_1 = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    handle_2 = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    
    handle_2.wait()
    if mod == 0 or mod == 3:
        Cij = torch.matmul(Xij, Wij.mT)
    elif mod == 1:
        Cij = torch.matmul(Xij, w_recv_list[3].mT) 
    else:
        Cij = torch.matmul(Xij, w_recv_list[0].mT) 
    
    handle_1.wait()
    
    if mod == 0:
        Cij = Cij + torch.matmul(x_recv_list[1], w_recv_list[1].mT) 
    elif mod == 1:
        Cij = Cij + torch.matmul(x_recv_list[0], w_recv_list[2].mT) 
    elif mod == 2:
        Cij = Cij + torch.matmul(x_recv_list[3], w_recv_list[1].mT) 
    elif mod == 3:
        Cij = Cij +  torch.matmul(x_recv_list[2], w_recv_list[2].mT) 

    return Cij
    
def XW_fourway(Xij, Wij, rank,  device, model_group=None):
    
    """
    Batched matrix-matrix (XW) multiplication distributed across four processes.
    
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
    torch.Tensor
        Local result.
    """

    mod = rank % 4
    
    empty = torch.empty(0, device=device)

    if mod == 0:
        x_send_list = [empty, Xij, empty, empty]
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty, empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, empty, get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1), empty]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty]
        x_recv_list = [get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij]
        w_recv_list = [empty, empty, empty, get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1)]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0)]
        w_send_list = [Wij, empty, empty, empty]       
        w_recv_list = [get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1), empty, empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty,  get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1), empty, empty]
        
    w_handle = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    x_handle = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    w_handle.wait()
    if mod == 0:
        Cij = torch.matmul(Xij, Wij)
    elif mod == 1:
        Cij = torch.matmul(Xij, w_recv_list[3]) 
    elif mod == 2:
        Cij = torch.matmul(Xij, w_recv_list[0]) 
    elif mod == 3:
        Cij =  torch.matmul(Xij, Wij) 
    x_handle.wait()
    if mod == 0:
        Cij = Cij + torch.matmul(x_recv_list[1], w_recv_list[2]) 
    elif mod == 1:
        Cij = Cij + torch.matmul(x_recv_list[0], Wij)
    elif mod == 2:
        Cij = Cij + torch.matmul(x_recv_list[3], Wij) 
    else:
        Cij = Cij + torch.matmul(x_recv_list[2], w_recv_list[1]) 
    return Cij

def XTW_fourway(Xij, Wij, rank, device, model_group=None):
    """
    Batched matrix-matrix (XTW) multiplication distributed across four processes.
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

    mod = rank % 4
    
    empty = torch.empty(0, device=device)

    if mod == 0:
        x_send_list = [empty, Xij, empty, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, empty, get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1), empty]
    elif mod == 1:
        x_send_list = [empty, empty, Xij, Xij]
        x_recv_list = [get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 1), empty]
        w_send_list = [empty, empty, empty, Wij]
        w_recv_list = [empty,  empty, empty, get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 2)]
    elif mod == 2: 
        x_send_list = [Xij, Xij, empty, empty]
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 1)]
        w_send_list = [Wij, empty, empty, empty]       
        w_recv_list = [get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 2), empty, empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, torch.float32, device, rank, 0), empty, empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty, get_preallocated_buffer(Wij.shape, torch.float32, device, rank, 1), empty, empty]
        
    x_handle = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    w_handle = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
        
    x_handle.wait()
    
    if mod == 0:
        Cij = torch.matmul(Xij.mT, Wij) 
    elif mod == 1:
        Cij = torch.matmul(x_recv_list[0].mT, Wij)
    elif mod == 2:
        Cij = torch.matmul(x_recv_list[3].mT, Wij) 
    elif mod == 3:
        Cij = torch.matmul(Xij.mT, Wij) 
    
    w_handle.wait()

    if mod == 0:
        Cij = Cij + torch.matmul(x_recv_list[2].mT, w_recv_list[2]) 
    elif mod == 1:
        Cij = Cij + torch.matmul(x_recv_list[2].mT, w_recv_list[3]) 
    elif mod == 2:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[0])
    elif mod == 3:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[1])
    
    return Cij


def XWT_eightway(Xij, Wij, rank, device, model_group=None):
    
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
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
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4)
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4)
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3)
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, Xij, empty]
        x_recv_list = [empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    Cij = get_preallocated_buffer((*Xij.shape[:-1], 2*Wij.shape[-2]), dtype=Xij.dtype, device=device, rank=rank, index=10) # new buffer...

    W_dim = Wij.shape[-2] 
    handle_w.wait()
    
    if mod == 0:
        Cij[:,:,:W_dim] = torch.matmul(Xij, Wij.mT)
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[2].mT)
    elif mod == 1:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[5].mT) 
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 2:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[0].mT)
        Cij[:,:,W_dim:] = torch.matmul(Xij, Wij.mT) 
    elif mod == 3:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[5].mT) 
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 4:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[0].mT)
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[2].mT)
    elif mod == 5:
        Cij[:,:,:W_dim] = torch.matmul(Xij, Wij.mT) 
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[7].mT)
    elif mod == 6:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[0].mT) 
        Cij[:,:,W_dim:] = torch.matmul(Xij, w_recv_list[2].mT)
    elif mod == 7:
        Cij[:,:,:W_dim] = torch.matmul(Xij, w_recv_list[5].mT)
        Cij[:,:,W_dim:] = torch.matmul(Xij, Wij.mT)
    
    handle_x.wait()
    
    if mod == 0:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[1], w_recv_list[1].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[1], w_recv_list[3].mT)
    elif mod == 1:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[0], w_recv_list[4].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[0], w_recv_list[6].mT)
    elif mod == 2:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[3], w_recv_list[1].mT) 
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[3], w_recv_list[3].mT)
    elif mod == 3:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[2], w_recv_list[4].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[2], w_recv_list[6].mT) 
    elif mod == 4:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[5], w_recv_list[1].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[5], w_recv_list[3].mT)
    elif mod == 5:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[4], w_recv_list[4].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[4], w_recv_list[6].mT)
    elif mod == 6:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[7], w_recv_list[1].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[7], w_recv_list[3].mT)
    elif mod == 7:
        Cij[:,:,:W_dim] += torch.matmul(x_recv_list[6], w_recv_list[4].mT)
        Cij[:,:,W_dim:] += torch.matmul(x_recv_list[6], w_recv_list[6].mT)

    return Cij

def XW_eightway_(Xij, Wij, rank,  device, model_group=None):
    
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
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty,
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1),
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3)
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1),
            empty,
            empty,
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2), 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            empty, 
            empty,
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3)
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2),
            empty, 
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            empty,
            empty,
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3)
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 2),
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty,
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, Xij, empty]
        x_recv_list = [empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1),
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
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

def XW_eightway(Xij, Wij, rank,  device, model_group=None):
    
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
    W0 = get_preallocated_buffer((*Wij.shape[:-2], Wij.shape[-2]*2, Wij.shape[-1]), dtype=Wij.dtype, device=device, rank=rank, index=1)
    W1 = get_preallocated_buffer((*Wij.shape[:-2], Wij.shape[-2]*2, Wij.shape[-1]), dtype=Wij.dtype, device=device, rank=rank, index=2)
    W0_first_half, W0_second_half = torch.chunk(W0, chunks=2, dim=-2)
    W1_first_half, W1_second_half = torch.chunk(W1, chunks=2, dim=-2)
    
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        W0_first_half.copy_(Wij)
        w_recv_list = [
            empty,
            empty,
            W0_second_half,
            empty, 
            W0_first_half, 
            empty,
            W1_second_half, 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        W1_first_half = W1_first_half + Wij
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            W1_second_half,
            empty,
            W0_first_half,
            empty,
            W0_second_half
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        W0_second_half.copy_(Wij)
        w_recv_list = [
            W0_first_half,
            empty,
            empty,
            empty, 
            W1_first_half, 
            empty,
            W1_second_half, 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty, empty, empty]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        W1_second_half.copy_(Wij)
        w_recv_list = [
            empty, 
            W1_first_half,
            empty, 
            empty,
            empty,
            W0_first_half,
            empty,
            W0_second_half
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        W1_first_half = W1_first_half + Wij
        w_recv_list = [
            W0_first_half,
            empty,
            W0_second_half,
            empty, 
            empty, 
            empty,
            W1_second_half, 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        W0_first_half.copy_(Wij)
        w_recv_list = [
            empty, 
            W1_first_half,
            empty, 
            W1_second_half,
            empty,
            empty,
            empty,
            W0_second_half
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        W1_second_half = W1_second_half + Wij
        w_recv_list = [
            W0_first_half,
            empty,
            W0_second_half,
            empty, 
            W1_first_half, 
            empty,
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, Xij, empty]
        x_recv_list = [empty, empty, empty, empty, empty, empty, get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0), empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        W0_second_half.copy_(Wij)
        w_recv_list = [
            empty, 
            W1_first_half,
            empty, 
            W1_second_half,
            empty,
            W0_first_half,
            empty,
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    handle_w.wait()
    
    Cij = torch.matmul(Xij, W0)
    
    handle_x.wait()
    
    if mod == 0:
        Cij = Cij + torch.matmul(x_recv_list[1], W1)
    elif mod == 1:
        Cij = Cij + torch.matmul(x_recv_list[0], W1)
    elif mod == 2:
        Cij = Cij + torch.matmul(x_recv_list[3], W1)
    elif mod == 3:
        Cij = Cij + torch.matmul(x_recv_list[2], W1)
    elif mod == 4:
        Cij = Cij + torch.matmul(x_recv_list[5], W1)
    elif mod == 5:
        Cij = Cij + torch.matmul(x_recv_list[4], W1)
    elif mod == 6:
        Cij = Cij + torch.matmul(x_recv_list[7], W1)
    elif mod == 7:
        Cij = Cij + torch.matmul(x_recv_list[6], W1)
    
    return Cij

def get_stacked_transpose(recv_list, indices):
    return torch.stack([recv_list[i].mT for i in indices], dim=1).contiguous()  # shape [4, 3, 5000, 2500]
    

def get_stacked(recv_list, indices):
    return torch.stack([recv_list[i] for i in indices], dim=1).contiguous()  # shape [4, 3, 5000, 2500]
    

def XTW_eightway(Xij, Wij, rank,  device, model_group=None):
    
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
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

    X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # split upon last dimension
    X0, X1 = X0.contiguous(), X1.contiguous()
    
    if mod == 0:
        x_send_list = [empty, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0), 
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty
        ]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 5), 
            empty
        ]
    elif mod == 1:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0), 
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1), 
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 3),
            empty
        ]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 5),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 6)
        ]
    elif mod == 2:
        x_send_list = [X0, X0, empty, X1, empty, empty, empty, empty]
        x_recv_list = [
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0), 
            empty,
            empty,
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty
        ]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 3), 
            empty,
            empty,
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 5), 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0), 
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1), 
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 3),
            empty
        ]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4),
            empty, 
            empty,
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 5),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 6)
        ]
    elif mod == 4:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 5), 
            empty, 
            empty,
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 6), 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, X0, empty, X1, X1]
        x_recv_list = [
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            empty,
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4),
            empty,
            empty,
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 5)
        ]
    elif mod == 6:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        w_recv_list = [
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 4), 
            empty,
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 5), 
            empty, 
            get_preallocated_buffer(Wij.shape, Xij.dtype, device, rank, 6), 
            empty, 
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, empty]
        x_recv_list = [
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 0),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 1),
            empty,
            get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, 2),
            empty,
            empty
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 3),
            empty, 
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 4),
            empty,
            get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 5),
            empty,
            empty
        ]

    x_handle = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    w_handle = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    
    x_handle.wait()

    if mod == 0:
        Cij = torch.matmul(X0.mT, Wij)
    elif mod == 1:
        Cij = torch.matmul(x_recv_list[0].mT, Wij)
    elif mod == 2:
        Cij = torch.matmul(X1.mT, Wij)
    elif mod == 3:
        Cij = torch.matmul(x_recv_list[2].mT, Wij)
    elif mod == 4:
        Cij = torch.matmul(x_recv_list[5].mT, Wij)
    elif mod == 5:
        Cij = torch.matmul(X0.mT, Wij)
    elif mod == 6:
        Cij = torch.matmul(x_recv_list[7].mT, Wij)
    elif mod == 7:
        Cij = torch.matmul(X1.mT, Wij)
    
    w_handle.wait()
    
    if mod == 0:
        Cij = Cij + torch.matmul(x_recv_list[2].mT, w_recv_list[2]) + torch.matmul(x_recv_list[4].mT, w_recv_list[4]) + torch.matmul(x_recv_list[6].mT, w_recv_list[6])
    elif mod == 1:
        Cij = Cij + torch.matmul(x_recv_list[2].mT, w_recv_list[3]) + torch.matmul(x_recv_list[4].mT, w_recv_list[5]) + torch.matmul(x_recv_list[6].mT, w_recv_list[7])
    elif mod == 2:
        Cij = Cij + torch.matmul(x_recv_list[0].mT, w_recv_list[0]) + torch.matmul(x_recv_list[4].mT, w_recv_list[4]) + torch.matmul(x_recv_list[6].mT, w_recv_list[6])
    elif mod == 3:
        Cij = Cij + torch.matmul(x_recv_list[0].mT, w_recv_list[1]) + torch.matmul(x_recv_list[4].mT, w_recv_list[5]) + torch.matmul(x_recv_list[6].mT, w_recv_list[7])
    elif mod == 4:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[0]) + torch.matmul(x_recv_list[3].mT, w_recv_list[2]) + torch.matmul(x_recv_list[7].mT, w_recv_list[6])
    elif mod == 5:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[1]) + torch.matmul(x_recv_list[3].mT, w_recv_list[3]) + torch.matmul(x_recv_list[7].mT, w_recv_list[7])
    elif mod == 6:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[0]) + torch.matmul(x_recv_list[3].mT, w_recv_list[2]) + torch.matmul(x_recv_list[5].mT, w_recv_list[4])
    elif mod == 7:
        Cij = Cij + torch.matmul(x_recv_list[1].mT, w_recv_list[1]) + torch.matmul(x_recv_list[3].mT, w_recv_list[3]) + torch.matmul(x_recv_list[5].mT, w_recv_list[5])
    
    return Cij

def XT_eightway(Xij, rank, device, model_group=None):
    mod = rank % 8
    
    empty = torch.empty(0, device=device, dtype=Xij.dtype)

    X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # split upon last dimension
    X0, X1 = X0.contiguous(), X1.contiguous()
    Cij0 = get_preallocated_buffer(X0.shape, Xij.dtype, device, rank, index=0)
    Cij1 = get_preallocated_buffer(X1.shape, Xij.dtype, device, rank, index=0)
    
    if mod == 0:
        x_send_list = [empty, X1, empty, empty, empty, empty, empty, empty]
        Cij0.copy_(X0)
        x_recv_list = [
            empty,
            empty,
            Cij1, 
            empty,
            empty,
            empty,
            empty,
            empty
        ]
    elif mod == 1:
        x_send_list = [empty, empty, empty, empty, X0, empty, X1, empty]
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            Cij0,
            empty,
            Cij1,
            empty
        ]
    elif mod == 2:
        Cij1.copy_(X1)
        x_send_list = [X0, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [
            Cij0,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, empty, empty, X0, empty, X1, empty]
        x_recv_list = [
            empty, 
            empty,
            empty,
            empty,
            Cij0,
            empty,
            Cij1,
            empty
        ]
    elif mod == 4:
        x_send_list = [empty, X0, empty, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            Cij0,
            empty,
            Cij1,
            empty,
            empty,
            empty,
            empty,
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, X1]
        Cij0.copy_(X0)
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            Cij1,
        ]
    elif mod == 6:
        x_send_list = [empty, X0, empty, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            Cij0,
            empty,
            Cij1,
            empty,
            empty,
            empty,
            empty,
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, X0, empty, empty]
        Cij1.copy_(X1)
        x_recv_list = [
            empty,
            empty,
            empty,
            empty,
            empty,
            Cij0,
            empty,
            empty
        ]

    dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=False)
    Cij = torch.concat([Cij0, Cij1], dim=-1).mT
    return Cij