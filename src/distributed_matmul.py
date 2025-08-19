import os

import torch
import torch.distributed as dist

streams = [torch.cuda.Stream() for _ in range(2)]
streamz = [torch.cuda.Stream() for _ in range(4)]

_prealloc_buffers = {}   
device = f'cuda:{int(os.getenv("SLURM_LOCALID"))}'
empty = torch.empty(0, device=device, dtype=torch.float32)
    
def _get_preallocated_buffer(
        shape: torch.Size, 
        dtype: torch.dtype, 
        rank: int, 
        device: torch.device, 
        index: int = 0
    ) -> torch.tensor:
    """
    Retrieve or create a preallocated buffer.

    Parameters.
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
def xw_transpose_twoway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int,
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
    bufferXW = _get_preallocated_buffer((*Xij.shape[:-1], W0.shape[-2]), torch.float32, rank, device) 

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

def xw_twoway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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

    empty = torch.empty(0, device=device)

    if mod == 0: 
        xw_send_list = [empty, Xij]
        xw_recv_list = [empty, bufferX]
                
    elif mod == 1:
        xw_send_list = [Xij, empty]
        xw_recv_list = [bufferX, empty]

    xw_handle = dist.all_to_all(xw_recv_list, xw_send_list, group=model_group, async_op=True)

    if mod == 0:
        Cij = torch.matmul(Xij, W0)
        xw_handle.wait()
        Cij = Cij + torch.matmul(xw_recv_list[1], W1)
    elif mod == 1:
        Cij = torch.matmul(Xij, W1)
        xw_handle.wait()
        Cij = Cij + torch.matmul(xw_recv_list[0], W0)

    
    return Cij 

def x_transpose_w_twoway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
    
    empty = torch.empty(0, device=device)

    if mod == 0: 
        xtw_send_list = [empty, Xij]
        xtw_recv_list  = [empty, bufferX] 
        
    elif mod == 1:
        xtw_send_list = [Xij, empty]
        xtw_recv_list = [bufferX, empty]
    
    xtw_handle = dist.all_to_all(xtw_recv_list, xtw_send_list, group=model_group, async_op=True)

    if mod == 0:
        Cij_top = torch.matmul(Xij.mT, Wij)
        xtw_handle.wait()
        Cij_bottom = torch.matmul(bufferX.mT, Wij)
    else:
        Cij_bottom = torch.matmul(Xij.mT, Wij)
        xtw_handle.wait()
        Cij_top = torch.matmul(bufferX.mT, Wij)
    
    Cij = torch.concat([Cij_top, Cij_bottom], dim=-2)
    return Cij


def x_transpose_twoway(
    X: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched transpose (XT) operation distributed across two processes.

    Parameters.
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
    bufferX = _get_preallocated_buffer(X1.shape, X1.dtype, rank, device) 
    empty = torch.empty(0, device=device)
    
    if mod == 0: 
        x_send_list = [empty, X1]
        x_recv_list = [empty, bufferX]
    elif mod == 1:
        x_send_list = [X0, empty]
        x_recv_list = [bufferX, empty]
    
    xt_handle = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
        
    if mod == 0:
        X0 = X0.mT    
        if len(X0.shape) == 3:
            XT = torch.empty(X0.shape[0], X0.shape[1]*2, X0.shape[2], device=device)
            XT[:, :X0.shape[1]] = X0
        else:
            XT = torch.empty(X0.shape[0]*2, X0.shape[1], device=device)
            XT[:X0.shape[1]] = X0
        xt_handle.wait()
        if len(X0.shape) == 3:
            XT[:, X0.shape[1]:, :] = bufferX.mT
        else:
            XT[X0.shape[1]:, :] = bufferX.mT
    elif mod == 1:
        X1 = X1.mT
        if len(X0.shape) == 3:
            XT = torch.empty(X1.shape[0], X1.shape[1]*2, X1.shape[2], device=device)
            XT[:,X1.shape[1]:] = X1
        else:
            XT = torch.empty(X1.shape[1]*2, X1.shape[2], device=device)
            XT[X1.shape[1]:] = X1
        xt_handle.wait()
        if len(X1.shape) == 3:
            XT[:, :X1.shape[1], :] = bufferX.mT
        else:
            XT[:X1.shape[1], :] = bufferX.mT
    return XT


def xw_transpose_fourway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), empty, empty]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty]       
        w_recv_list = [empty, empty, _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1), _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2)]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0)]
        w_send_list = [empty, Wij, empty, Wij]
        w_recv_list = [_get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1), _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2), empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty, empty, _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), empty]

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
        Cij = Cij + torch.matmul(x_recv_list[2], w_recv_list[2].mT) 

    return Cij
    
def xw_fourway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int,
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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

    xdtype, wdtype = Xij.dtype, Wij.dtype
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, xdtype, rank, device, 0), empty, empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, empty, _get_preallocated_buffer(Wij.shape, wdtype, rank, device, 1), empty]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, xdtype, rank, device, 0), empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij]
        w_recv_list = [empty, empty, empty, _get_preallocated_buffer(Wij.shape, wdtype, rank, device, 1)]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, xdtype, rank, device, 0)]
        w_send_list = [Wij, empty, empty, empty]       
        w_recv_list = [_get_preallocated_buffer(Wij.shape, wdtype, rank, device, 1), empty, empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, xdtype, rank, device, 0), empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty,  _get_preallocated_buffer(Wij.shape, wdtype, rank, device, 1), empty, empty]
        
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

def x_transpose_w_fourway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
    model_group : torch.distributed.ProcessGroup
        Process group for an individual model-parallel instance.

    Returns
    -------
    Cij : torch.Tensor
        Local result.
    """
    mod = rank % 4
    
    empty = torch.empty(0, device=device)

    xdtype, wdtype = Xij.dtype, Wij.dtype
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, xdtype, device, rank, 0), empty]
        w_send_list = [empty, empty, Wij, empty]       
        w_recv_list = [empty, empty, _get_preallocated_buffer(Wij.shape, wdtype, device, rank, 1), empty]
    elif mod == 1:
        x_send_list = [empty, empty, Xij, Xij]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, xdtype, device, rank, 0), empty, _get_preallocated_buffer(Xij.shape, xdtype, device, rank, 1), empty]
        w_send_list = [empty, empty, empty, Wij]
        w_recv_list = [empty,  empty, empty, _get_preallocated_buffer(Wij.shape, wdtype, device, rank, 2)]
    elif mod == 2: 
        x_send_list = [Xij, Xij, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, xdtype, device, rank, 0), empty, _get_preallocated_buffer(Xij.shape, xdtype, device, rank, 1)]
        w_send_list = [Wij, empty, empty, empty]       
        w_recv_list = [_get_preallocated_buffer(Wij.shape, wdtype, device, rank, 2), empty, empty, empty]
    else:
        x_send_list = [empty, empty, Xij, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, xdtype, device, rank, 0), empty, empty]
        w_send_list = [empty, Wij, empty, empty]
        w_recv_list = [empty, _get_preallocated_buffer(Wij.shape, wdtype, device, rank, 1), empty, empty]
        
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

def x_transpose_fourway(
    X: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched transpose (XT) operation distributed across four processes.

    Parameters.
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
        x_recv_list = [empty, empty, _get_preallocated_buffer(X.shape, X.dtype, device, rank, 0), empty]
    else: # mod == 2
        x_send_list = [empty,  X, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(X.shape, X.dtype, device, rank, 0), empty, empty]

    dist.all_to_all(x_recv_list, x_send_list, group=model_group)
    
    if mod == 0 or mod == 3:
        return X.mT
    elif mod == 1:
        return x_recv_list[2].mT
    else:
        return x_recv_list[1].mT
    
def xw_transpose_eightway_(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
    
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4)
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4)
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3)
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0)]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
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
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    Cij = _get_preallocated_buffer((*Xij.shape[:-1], 2*Wij.shape[-2]), dtype=Xij.dtype, rank=rank, device=device, index=10) # new buffer...

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
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[1], w_recv_list[1].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[1], w_recv_list[3].mT)
    elif mod == 1:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[0], w_recv_list[4].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[0], w_recv_list[6].mT)
    elif mod == 2:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[3], w_recv_list[1].mT) 
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[3], w_recv_list[3].mT)
    elif mod == 3:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[2], w_recv_list[4].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[2], w_recv_list[6].mT) 
    elif mod == 4:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[5], w_recv_list[1].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[5], w_recv_list[3].mT)
    elif mod == 5:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[4], w_recv_list[4].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[4], w_recv_list[6].mT)
    elif mod == 6:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[7], w_recv_list[1].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[7], w_recv_list[3].mT)
    elif mod == 7:
        Cij[:,:,:W_dim] = Cij[:,:,:W_dim] + torch.matmul(x_recv_list[6], w_recv_list[4].mT)
        Cij[:,:,W_dim:] = Cij[:,:,W_dim:] + torch.matmul(x_recv_list[6], w_recv_list[6].mT)


    return Cij

def xw_transpose_eightway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.

    Parameters.
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
    
    #empty = torch.empty(0, device=device, dtype=Xij.dtype)
    
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4)
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4)
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            empty,
            empty, 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3)
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0)]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 1), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 2), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
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
            empty, 
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 1),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 2),
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    Cij0 = _get_preallocated_buffer((*Xij.shape[:-1], Wij.shape[-2]), dtype=Xij.dtype, rank=rank, device=device, index=10) # new buffer...
    Cij1 = _get_preallocated_buffer((*Xij.shape[:-1], Wij.shape[-2]), dtype=Xij.dtype, rank=rank, device=device, index=11) # new buffer...

    with torch.cuda.stream(streams[0]):
        handle_w.wait()
    with torch.cuda.stream(streams[1]):
        handle_w.wait()  
        
    W_dim = Wij.shape[-2] 

    if mod == 0:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, Wij.mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[2].mT, out=Cij1)
    elif mod == 1:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[5].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[7].mT, out=Cij1)
    elif mod == 2:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[0].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, Wij.mT, out=Cij1)
    elif mod == 3:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[5].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[7].mT, out=Cij1)
    elif mod == 4:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[0].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[2].mT, out=Cij1)
    elif mod == 5:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, Wij.mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[7].mT, out=Cij1)
    elif mod == 6:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[0].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, w_recv_list[2].mT, out=Cij1)
    elif mod == 7:
        with torch.cuda.stream(streams[0]):
            torch.matmul(Xij, w_recv_list[5].mT, out=Cij0)
        with torch.cuda.stream(streams[1]):
            torch.matmul(Xij, Wij.mT, out=Cij1)

    with torch.cuda.stream(streams[0]):
        handle_x.wait()
    with torch.cuda.stream(streams[1]):
        handle_x.wait()  
    
    if mod == 0:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[1], w_recv_list[1].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[1], w_recv_list[3].mT)
    elif mod == 1:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[0], w_recv_list[4].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[0], w_recv_list[6].mT)
    elif mod == 2:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[3], w_recv_list[1].mT) 
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[3], w_recv_list[3].mT)
    elif mod == 3:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[2], w_recv_list[4].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[2], w_recv_list[6].mT) 
    elif mod == 4:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[5], w_recv_list[1].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[5], w_recv_list[3].mT)
    elif mod == 5:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[4], w_recv_list[4].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[4], w_recv_list[6].mT)
    elif mod == 6:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[7], w_recv_list[1].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[7], w_recv_list[3].mT)
    elif mod == 7:
        with torch.cuda.stream(streams[0]):
            Cij0 = Cij0 + torch.matmul(x_recv_list[6], w_recv_list[4].mT)
        with torch.cuda.stream(streams[1]):
            Cij1 = Cij1 + torch.matmul(x_recv_list[6], w_recv_list[6].mT)
    
    for s in streams:
        s.synchronize()

    Cij = torch.concat([Cij0, Cij1], dim=-1)

    return Cij

def xw_eightway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int,  
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
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
        ws_stacked = torch.concatenate([w_recv_list[5], Wij], dim=0).contiguous() 
        #Cij = torch.matmul(X0, w_recv_list[5]) + torch.matmul(X1, Wij)  
    
    Cij = torch.matmul(Xij, ws_stacked)
    if rank == 0:
        print(f'Xij: {Xij.shape}, {Wij.shape}, {ws_stacked.shape}')
    
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

def xw_eightway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched matrix-matrix (XW) multiplication distributed across eight processes.
    
    Parameters.
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
    
    #empty = torch.empty(0, device=device, dtype=Xij.dtype)
    W0 = _get_preallocated_buffer((*Wij.shape[:-2], Wij.shape[-2]*2, Wij.shape[-1]), dtype=Wij.dtype, rank=rank, device=device, index=1)
    W1 = _get_preallocated_buffer((*Wij.shape[:-2], Wij.shape[-2]*2, Wij.shape[-1]), dtype=Wij.dtype, rank=rank, device=device, index=2)
    
    W0_shape = W0.shape
    split_size = W0_shape[-2]//2
    if mod == 0:
        x_send_list = [empty, Xij, empty, empty, empty, empty, empty, empty]
        x_recv_list = [empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        W0[..., :split_size, :].copy_(Wij)
        w_recv_list = [
            empty,
            empty,
            W0[..., split_size:, :],
            empty, 
            W1[..., :split_size, :], 
            empty,
            W1[..., split_size:, ], 
            empty
        ]
    elif mod == 1:
        x_send_list = [Xij, empty, empty, empty, empty, empty, empty, empty]
        x_recv_list = [_get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty, empty, empty]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        W1[..., :split_size, :].copy_(Wij)
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            W1[..., split_size:, ],
            empty,
            W0[..., :split_size, :],
            empty,
            W0[..., split_size:, ]
        ]
    elif mod == 2:
        x_send_list = [empty, empty, empty, Xij, empty, empty, empty, empty]
        x_recv_list = [empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        W0[..., split_size:, ].copy_(Wij)
        w_recv_list = [
            W0[..., :split_size, :],
            empty,
            empty,
            empty, 
            W1[..., :split_size, :], 
            empty,
            W1[..., split_size:, ], 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, Xij, empty, empty, empty, empty, empty]
        x_recv_list = [empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty, empty, empty]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        W0[W0_shape[0]//2:].copy_(Wij)
        w_recv_list = [
            empty, 
            W1[..., :split_size, :],
            empty, 
            empty,
            empty,
            W0[..., :split_size, :],
            empty,
            W0[W0_shape[0]//2:]
        ]
    elif mod == 4:
        x_send_list = [empty, empty, empty, empty, empty, Xij, empty, empty]
        x_recv_list = [empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        W1[..., :split_size, :].copy_(Wij)
        w_recv_list = [
            W0[..., :split_size, :],
            empty,
            W0[W0_shape[0]//2:],
            empty, 
            empty, 
            empty,
            W1[W0_shape[0]//2:], 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, Xij, empty, empty, empty]
        x_recv_list = [empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty, empty, empty]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        W0[..., :split_size, :].copy_(Wij)
        w_recv_list = [
            empty, 
            W1[..., :split_size, :],
            empty, 
            W1[W0_shape[0]//2:],
            empty,
            empty,
            empty,
            W0[W0_shape[0]//2:]
        ]
    elif mod == 6:
        x_send_list = [empty, empty, empty, empty, empty, empty, empty, Xij]
        x_recv_list = [empty, empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0)]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        W1[W0_shape[0]//2:].copy_(Wij)
        w_recv_list = [
            W0[..., :split_size, :],
            empty,
            W0[W0_shape[0]//2:],
            empty, 
            W1[..., :split_size, :], 
            empty,
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, empty, empty, Xij, empty]
        x_recv_list = [empty, empty, empty, empty, empty, empty, _get_preallocated_buffer(Xij.shape, Xij.dtype, rank, device, 0), empty]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        W0[W0_shape[0]//2:].copy_(Wij)
        w_recv_list = [
            empty, 
            W1[..., :split_size, :],
            empty, 
            W1[W0_shape[0]//2:],
            empty,
            W0[..., :split_size, :],
            empty,
            empty
        ]

    handle_w = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    handle_x = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    
    with torch.cuda.stream(streams[0]):
        handle_w.wait()
        Cij0 = torch.matmul(Xij, W0)
    
    with torch.cuda.stream(streams[0]):
        handle_w.wait()
        handle_x.wait()
        
        if mod == 0:
            Cij1 = torch.matmul(x_recv_list[1], W1)
        elif mod == 1:
            Cij1 = torch.matmul(x_recv_list[0], W1)
        elif mod == 2:
            Cij1 = torch.matmul(x_recv_list[3], W1)
        elif mod == 3:
            Cij1 = torch.matmul(x_recv_list[2], W1)
        elif mod == 4:
            Cij1 = torch.matmul(x_recv_list[5], W1)
        elif mod == 5:
            Cij1 = torch.matmul(x_recv_list[4], W1)
        elif mod == 6:
            Cij1 = torch.matmul(x_recv_list[7], W1)
        elif mod == 7:
            Cij1 = torch.matmul(x_recv_list[6], W1)

    
    for s in streams:
        s.synchronize()
    
    Cij = Cij0 + Cij1
    
    return Cij

def x_transpose_w_eightway_(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched matrix-matrix (XWT) multiplication distributed across four processes.
    
    Parameters.
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
    
    X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # split upon last dimension
    X0, X1 = X0.contiguous(), X1.contiguous()
    
    if mod == 0:
        x_send_list = [empty, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty
        ]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty
        ]
    elif mod == 1:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1), 
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 3),
            empty
        ]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 6)
        ]
    elif mod == 2:
        x_send_list = [X0, X0, empty, X1, empty, empty, empty, empty]
        x_recv_list = [
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0), 
            empty,
            empty,
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty
        ]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty,
            empty,
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1), 
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 3),
            empty
        ]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty, 
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 6)
        ]
    elif mod == 4:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty, 
            empty,
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 6), 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, X0, empty, X1, X1]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5)
        ]
    elif mod == 6:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 6), 
            empty, 
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, 2),
            empty,
            empty
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
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


def x_transpose_w_eightway(
    Xij: torch.tensor, 
    Wij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    """
    Batched matrix-matrix (XTW) multiplication distributed across eight processes.

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
    
    X0, X1 = torch.split(Xij.mT, Xij.shape[-1]//2, dim=-2)
    X0, X1 = X0.contiguous(), X1.contiguous()
    
    Xmt_shape = X0.shape
    if mod == 0:
        x_send_list = [empty, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty
        ]
        w_send_list = [empty, empty, Wij, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            empty, 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty
        ]
    elif mod == 1:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1), 
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 3),
            empty
        ]
        w_send_list = [empty, empty, empty, Wij, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            empty, 
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 6)
        ]
    elif mod == 2:
        x_send_list = [X0, X0, empty, X1, empty, empty, empty, empty]
        x_recv_list = [
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0), 
            empty,
            empty,
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty
        ]
        w_send_list = [Wij, empty, empty, empty, Wij, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 3), 
            empty,
            empty,
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty
        ]
    elif mod == 3:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, X1]
        x_recv_list = [
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0), 
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1), 
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 3),
            empty
        ]
        w_send_list = [empty, Wij, empty, empty, empty, Wij, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty, 
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 6)
        ]
    elif mod == 4:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, empty, empty, Wij, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty, 
            empty,
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 6), 
            empty
        ]
    elif mod == 5:
        x_send_list = [empty, empty, empty, empty, X0, empty, X1, X1]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, empty, empty, Wij]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            empty,
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5)
        ]
    elif mod == 6:
        x_send_list = [X0, X0, X1, X1, empty, empty, empty, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 3),
        ]
        w_send_list = [Wij, empty, Wij, empty, Wij, empty, empty, empty]
        w_recv_list = [
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 4), 
            empty,
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 5), 
            empty, 
            _get_preallocated_buffer(Wij.shape, Xij.dtype, rank, device, 6), 
            empty, 
            empty,
            empty
        ]
    elif mod == 7:
        x_send_list = [empty, empty, empty, empty, X0, X0, X1, empty]
        x_recv_list = [
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 0),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 1),
            empty,
            _get_preallocated_buffer(Xmt_shape, Xij.dtype, rank, device, 2),
            empty,
            empty
        ]
        w_send_list = [empty, Wij, empty, Wij, empty, Wij, empty, empty]
        w_recv_list = [
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 3),
            empty, 
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 4),
            empty,
            _get_preallocated_buffer(Wij.shape, Wij.dtype, rank, device, 5),
            empty,
            empty
        ]

    x_handle = dist.all_to_all(x_recv_list, x_send_list, group=model_group, async_op=True)
    w_handle = dist.all_to_all(w_recv_list, w_send_list, group=model_group, async_op=True)
    
    with torch.cuda.stream(streamz[0]):
        x_handle.wait()
        if mod == 0:
            Cij0 = torch.matmul(X0, Wij)
        elif mod == 1:
            Cij0 = torch.matmul(x_recv_list[0], Wij)
        elif mod == 2:
            Cij0 = torch.matmul(X1, Wij)
        elif mod == 3:
            Cij0 = torch.matmul(x_recv_list[2], Wij)
        elif mod == 4:
            Cij0 = torch.matmul(x_recv_list[5], Wij)
        elif mod == 5:
            Cij0 = torch.matmul(X0, Wij)
        elif mod == 6:
            Cij0 = torch.matmul(x_recv_list[7], Wij)
        elif mod == 7:
            Cij0 = torch.matmul(X1, Wij)
    
    for stream in streamz[1:]:
        with torch.cuda.stream(stream):
            w_handle.wait()
    
    if mod == 0:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[2], w_recv_list[2])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[4], w_recv_list[4])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[6], w_recv_list[6])
    elif mod == 1:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[2], w_recv_list[3])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[4], w_recv_list[5])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[6], w_recv_list[7])
    elif mod == 2:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[0], w_recv_list[0])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[4], w_recv_list[4])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[6], w_recv_list[6])
    elif mod == 3:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[0], w_recv_list[1])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[4], w_recv_list[5])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[6], w_recv_list[7])
    elif mod == 4:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[1], w_recv_list[0])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[3], w_recv_list[2])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[7], w_recv_list[6])
    elif mod == 5:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[1], w_recv_list[1])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[3], w_recv_list[3])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[7], w_recv_list[7])
    elif mod == 6:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[1], w_recv_list[0])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[3], w_recv_list[2])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[5], w_recv_list[4])
    elif mod == 7:
        with torch.cuda.stream(streamz[1]):
            Cij1 = torch.matmul(x_recv_list[1], w_recv_list[1])
        with torch.cuda.stream(streamz[2]):
            Cij2 = torch.matmul(x_recv_list[3], w_recv_list[3])
        with torch.cuda.stream(streamz[3]):
            Cij3 = torch.matmul(x_recv_list[5], w_recv_list[5])
    
    for s in streamz:
        s.synchronize()

    Cij = Cij0 + Cij1 + Cij2 + Cij3
    
    return Cij

def x_transpose_eightway_(
    Xij: torch.tensor, 
    rank: int, 
    device: torch.device, 
    model_group: torch.distributed.ProcessGroup
) -> torch.tensor:
    mod = rank % 8
    
    X0, X1 = torch.split(Xij, Xij.shape[-1]//2, dim=-1) # split upon last dimension
    X0, X1 = X0.contiguous(), X1.contiguous()
    Cij0 = _get_preallocated_buffer(X0.shape, Xij.dtype, rank, device, index=0)
    Cij1 = _get_preallocated_buffer(X1.shape, Xij.dtype, rank, device, index=0)
    
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
