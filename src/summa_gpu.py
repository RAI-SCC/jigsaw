#from mpi4py import MPI
import numpy as np
import torch
import torch.distributed as dist
import time

import torch.distributed


def XW_fourway(Xij, Wij, rank,  device, model_group):
    
    """
    Perform non-blocking send and receive operations for distributed matrix multiplication.
    
    Parameters
    ----------
    Xij : torch.Tensor
        The local part of matrix X.
    Wij : torch.Tensor
        The local part of matrix W.
    rank : int
        The rank of the current process.
    device : torch.device
        The device (CPU/GPU) for tensors.
    model_group : torch.distributed.ProcessGroup
        The process group for communication.
    
    Returns
    -------
    torch.Tensor
        The result of the matrix multiplication.
    """

    mod = rank % 4
    bufferX = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)
    bufferW = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1)
    
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        req1 = dist.isend(tensor=Xij, dst=rank+1, group=model_group)
        req2 = dist.irecv(tensor=bufferX, src=rank+1, group=model_group)
        req3 = dist.isend(tensor=Wij, dst=rank+2, group=model_group)
        req4 = dist.irecv(tensor=bufferW, src=rank+2, group=model_group)
        Cij = torch.matmul(Xij, Wij)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()
        Cij = Cij + torch.matmul(bufferX, bufferW)   
    elif mod == 1:
        req1 = dist.irecv(tensor=bufferX, src=rank-1, group=model_group) 
        req2 = dist.isend(tensor=Xij, dst=rank-1, group=model_group)
        req1.wait()
        req2.wait()
        Cij = torch.matmul(bufferX, Wij)
        req3 = dist.isend(tensor=Wij, dst=rank+2, group=model_group)
        req4 = dist.irecv(tensor=bufferW, src=rank+2, group=model_group)
        req3.wait()
        req4.wait()
        Cij = Cij + torch.matmul(Xij, bufferW) 
    elif mod == 2:
        req1 = dist.isend(tensor=Xij, dst=rank+1, group=model_group)
        req2 = dist.irecv(tensor=bufferX, src=rank+1, group=model_group)
        req1.wait()
        req2.wait()
        Cij = torch.matmul(bufferX, Wij)
        req3 = dist.irecv(tensor=bufferW, src=rank-2, group=model_group)
        req4 = dist.isend(tensor=Wij, dst=rank-2, group=model_group)
        req4.wait()
        req3.wait()
        
        Cij = Cij + torch.matmul(Xij, bufferW)      
    else:
        req1 = dist.irecv(tensor=bufferX, src=rank-1, group=model_group)
        req2 = dist.isend(tensor=Xij, dst=rank-1, group=model_group)
        
        req3 = dist.irecv(tensor=bufferW, src=rank-2, group=model_group)
        req4 = dist.isend(tensor=Wij, dst=rank-2, group=model_group)
        Cij = torch.matmul(Xij, Wij)
        req1.wait()
        req2.wait()
        req4.wait()
        req3.wait()
        Cij = Cij + torch.matmul(bufferX, bufferW)   
        
    return Cij


def matrix_transpose(X, rank, device):
    """
    Transpose a matrix using distributed communication based on the rank of the process.
    Parameters
    ----------
    X : torch.Tensor
        The input matrix to be transposed.
    rank : int
        The rank of the current process in the distributed setup.
    device : torch.device
        The device on which the tensor is located.
    Returns
    -------
    torch.Tensor
        The transposed matrix.
    """

    mod = rank % 4
    if mod == 1:
        buffer = get_preallocated_buffer(X.shape, X.dtype, device, rank)
        send_transpose = torch.distributed.isend(X, dst=rank+1)
        recv_transpose = torch.distributed.irecv(buffer, src=rank+1)
        send_transpose.wait()
        recv_transpose.wait()
        X_T = buffer
    elif mod == 2:
        buffer = get_preallocated_buffer(X.shape, X.dtype, device, rank)
        recv_transpose = torch.distributed.irecv(buffer, src=rank-1)
        send_transpose = torch.distributed.isend(X, dst=rank-1)

        send_transpose.wait()
        recv_transpose.wait()
        
        X_T = buffer
    else:
        X_T = X

    return X_T.mT

    
def matrix_transpose_twoway(X, rank, device, model_group):
    """
    Perform a two-way distributed matrix transpose.
    
    Parameters
    ----------
    X : torch.Tensor
        Input tensor to transpose. Shape: [n_batch, a, b].
    rank : int
        Rank of the current process.
    device : torch.device
        Device for tensor operations.
    model_group : torch.distributed.ProcessGroup
        Process group for communication.
    
    Returns
    -------
    torch.Tensor
        Transposed tensor.
    """

    mod = rank % 2

    X0, X1 = torch.split(X, X.shape[-2]//2, dim=-2)
    # X: [n_batch, a, b] Xt = []
    #XT = torch.empty()

    bufferX = get_preallocated_buffer(X1.shape, torch.float32, device, rank) 
        
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
    
        send_B_1 = torch.distributed.isend(X1, dst=rank+1, group=None)
        recv_B_1 = torch.distributed.irecv(bufferX, src=rank+1, group=None)
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
        recv_transpose = torch.distributed.irecv(bufferX, src=rank-1, group=None)
        send_transpose = torch.distributed.isend(X0, dst=rank-1, group=None)
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



def XWT_fourway(Xij, Wij, rank,  device, model_group):
    
    """
    Perform precomputation and asynchronous send/receive operations for distributed XWT matrix multiplication.
    Parameters
    ----------
    Xij : torch.Tensor
        The input tensor X.
    Wij : torch.Tensor
        The input tensor W.
    rank : int
        The rank of the current process in the distributed group.
    device : torch.device
        The device (CPU/GPU) on which the tensors are allocated.
    model_group : torch.distributed.ProcessGroup
        The process group for collective communication.
    Returns
    -------
    Cij : torch.Tensor
        The result of the matrix multiplication and communication operations.
    """
    
    mod = rank % 4
    
    if mod == 0: 
        if len(Xij.shape) > 2:
            bufferXW = get_preallocated_buffer((*Xij.shape[:-1], Wij.shape[0]), Xij.dtype, device, rank)
        else:
            bufferXW = get_preallocated_buffer((Wij.shape[0], Xij.shape[0], Wij.shape[-2]), Xij.dtype, device, rank)
        
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        req_1 = dist.isend(Xij, dst=rank+1, group=model_group)
        req_2 = dist.isend(Wij, dst=rank+2, group=model_group)
        req_3 = dist.irecv(bufferXW, src=rank+1, group=model_group)
        Cij = torch.matmul(Xij, Wij.mT)
        req_1.wait()
        req_2.wait()
        req_3.wait()
        
        
    elif mod == 1:

        bufferX  = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)
        bufferW2 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1) 
        bufferW3 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2) 
        
        req_1 = dist.irecv(bufferX, src=rank-1, group=model_group)
        req_2 = dist.isend(Wij, dst=rank+1, group=model_group)
        req_3 = dist.irecv(bufferW2, src=rank+1, group=model_group)
        req_4 = dist.irecv(bufferW3, src=rank+2, group=model_group)
        
        XW = torch.matmul(Xij, Wij.mT)
        req_5 = dist.isend(XW, dst=rank-1, group=model_group)
        req_2.wait()
        req_5.wait()
        req_4.wait()
        Cij = torch.matmul(Xij, bufferW3.mT)
        req_1.wait()
        req_3.wait()
                          
    elif mod == 2:
        bufferX  = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)
        bufferW0 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 1) 
        bufferW1 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2) 
        req_1  = dist.irecv(tensor=bufferX,  src=rank+1, group=model_group)
        req_2 = dist.irecv(tensor=bufferW0, src=rank-2, group=model_group)
        req_3 = dist.irecv(tensor=bufferW1, src=rank-1, group=model_group)
        req_4 = dist.isend(Wij, dst=rank-1, group=model_group)
        
        XW = torch.matmul(Xij, Wij.mT)

        req_5 = dist.isend(tensor=XW, dst=rank+1, group=model_group)
        req_4.wait()
        req_5.wait()
        req_2.wait()
        Cij = torch.matmul(Xij, bufferW0.mT) 
        req_1.wait()
        req_3.wait()
        

    else:
        if len(Xij.shape) > 2:
            bufferXW = get_preallocated_buffer((*Xij.shape[:-1], Wij.shape[0]), Xij.dtype, device, rank)
        else:
            bufferXW = get_preallocated_buffer((Wij.shape[0], Xij.shape[0], Wij.shape[-2]), Xij.dtype, device, rank)
        
        req_1  = dist.isend(tensor=Xij, dst=rank-1, group=model_group)
        req_2  = dist.isend(tensor=Wij, dst=rank-2, group=model_group)
        req_3  = dist.irecv(tensor=bufferXW, src=rank-1, group=model_group)
        
        Cij   = torch.matmul(Xij, Wij.mT)
        
        req_1.wait()
        req_2.wait()
        req_3.wait()

    if mod in [0, 3]:
        Cij = Cij + bufferXW
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferX, bufferW2.mT)
    else:
        Cij = Cij + torch.matmul(bufferX, bufferW1.mT)
    
    return Cij

_prealloc_buffers = {}

def get_preallocated_buffer(shape, dtype, device, rank, index=0):
    """
    Retrieve or create a preallocated buffer with the specified shape, dtype, and device.
    Parameters
    ----------
    shape : tuple
        The shape of the buffer to be created or retrieved.
    dtype : torch.dtype
        The data type of the buffer.
    device : torch.device
        The device on which the buffer will be allocated.
    rank : int
        The rank of the process requesting the buffer.
    index : int, optional
        An additional index to differentiate buffers (default is 0).
    Returns
    -------
    torch.Tensor
        A tensor with the specified shape, dtype, and device, either retrieved from the preallocated buffers or newly created.
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


def XTW_fourway(Xij, Wij, rank, device, model_group):
    """
    Precomputes the product of Xij and Wij with custom communication patterns using non-blocking sends and receives. Calculate ATB matrix--matrix multiplication.
    Parameters
    ----------
    Xij : torch.Tensor
        The matrix Aij to be multiplied.
    Wij : torch.Tensor
        The matrix Bij to be multiplied.
    rank : int
        The rank of the current process in the distributed group.
    device : torch.device
        The device on which the tensors are allocated.
    model_group : torch.distributed.ProcessGroup
        The process group for collective communication.
    Returns
    -------
    torch.Tensor
        The resulting matrix Cij after the precomputation and communication.
    """

    mod = rank % 4
    
    if mod == 0: 
        if len(Xij.shape) > 2:
            bufferXW = get_preallocated_buffer((Xij.shape[0], Xij.shape[-1], Wij.shape[-1]), Xij.dtype, device, rank)
        else:
            bufferXW = get_preallocated_buffer(Wij.shape[0], Xij.shape[-1], Wij.shape[-1], rank)
        
        req_1 = dist.isend(tensor=Xij, dst=rank+1, group=model_group)
        req_2 = dist.isend(tensor=Wij, dst=rank+2, group=model_group)
        req_3 = dist.irecv(tensor=bufferXW, src=rank+2, group=model_group)
        
        Cij = torch.matmul(Xij.mT, Wij)

        req_1.wait()
        req_2.wait()
        req_3.wait()

    elif mod == 1:
        bufferX0 = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)
        bufferX2 = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 1)
        bufferW3 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2)
        
        req_1 = dist.irecv(tensor=bufferX0, src=rank-1, group=model_group)
        req_2 = dist.irecv(tensor=bufferX2, src=rank+1, group=model_group)
        req_3 = dist.isend(tensor=Xij, dst=rank+1, group=model_group)
        req_4 = dist.irecv(tensor=bufferW3, src=rank+2, group=model_group)

        XW = torch.matmul(Xij.mT, Wij)
        req_5 = dist.isend(tensor=XW, dst=rank+2, group=model_group)
        req_3.wait()
        req_5.wait()
        req_1.wait()
        Cij = torch.matmul(bufferX0.mT, Wij)
        req_2.wait()
        req_4.wait()
        
        
        
    elif mod == 2:
        bufferX1 = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 0)
        bufferX3 = get_preallocated_buffer(Xij.shape, Xij.dtype, device, rank, 1)
        bufferW0 = get_preallocated_buffer(Wij.shape, Wij.dtype, device, rank, 2)
        req_1 = dist.isend(tensor=Xij, dst=rank-1, group=model_group)
        req_2 = dist.irecv(tensor=bufferX1, src=rank-1, group=model_group)
        req_3 = dist.irecv(tensor=bufferW0, src=rank-2, group=model_group)
        req_4 = dist.irecv(tensor=bufferX3, src=rank+1, group=model_group)

        XW = torch.matmul(Xij.mT, Wij)
        req_5 = dist.isend(tensor=XW, dst=rank-2, group=model_group)

        req_1.wait()
        req_5.wait()
        req_4.wait()
        Cij = torch.matmul(bufferX3.mT, Wij) 
        req_2.wait()
        req_3.wait()
        
    else:
        if len(Xij.shape) > 2:
            bufferXW = get_preallocated_buffer((Xij.shape[0], Xij.shape[-1], Wij.shape[-1]), Xij.dtype, device, rank, 0)
        else:
            bufferXW = get_preallocated_buffer((Wij.shape[0], Xij.shape[-1], Wij.shape[-1]), Xij.dtype, device, rank, 0)
        
        req_1 = dist.isend(tensor=Xij, dst=rank-1, group=model_group)
        req_2 = dist.isend(tensor=Wij, dst=rank-2, group=model_group)
        req_3 = dist.irecv(tensor=bufferXW, src=rank-2, group=model_group)
        
        Cij = torch.matmul(Xij.mT, Wij)
        req_1.wait()
        req_2.wait()
        req_3.wait()

    if mod in [0, 3]:
        Cij = Cij + bufferXW
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferX2.mT, bufferW3) 
    else:
        Cij = Cij + torch.matmul(bufferX1.mT, bufferW0) 
    
    return Cij

def XW_twoway(Xij, Wij, device, rank, model_group=None):
    """
    Perform a two-way matrix multiplication with communication between ranks.
    Parameters
    ----------
    Xij : torch.Tensor
        The input matrix A.
    Wij : torch.Tensor
        The input matrix B.
    device : torch.device
        The device to perform computations on.
    rank : int
        The rank of the current process.
    model_group : torch.distributed.ProcessGroup, optional
        The process group for communication (default is None).
    Returns
    -------
    torch.Tensor
        The resulting matrix after the two-way multiplication and communication.
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


def XWT_twoway(Xij, Wij, device, rank, model_group=None):
    """
    Perform a two-way asynchronous block transfer (XWT) matrix multiplication.
    This function performs a two-way asynchronous block transfer matrix multiplication
    using PyTorch's distributed communication primitives. It splits the input matrix `Bij`
    into two parts, performs matrix multiplications, and exchanges the results between
    neighboring ranks.
    Parameters
    ----------
    Xij : torch.Tensor
        The input matrix X.
    Wij : torch.Tensor
        The input matrix W.
    device : torch.device
        The device on which the tensors are allocated.
    rank : int
        The rank of the current process in the distributed group.
    model_group : torch.distributed.ProcessGroup, optional
        The process group for communication. Default is None.
    Returns
    -------
    torch.Tensor
        The resulting matrix after performing the  matrix multiplication.
    """

    mod = rank % 2
    
    
    W0, W1 = torch.split(Wij, Wij.shape[-2]//2, dim=-2)
    
    bufferXW = get_preallocated_buffer((*Xij.shape[:-1], W0.shape[-2]), torch.float32, device, rank) 
    
    if mod == 0: 
        XW = torch.matmul(Xij, W1.mT)
        send_B_1 = torch.distributed.isend(XW, dst=rank+1, group=None)
        recv_B_1 = torch.distributed.irecv(bufferXW, src=rank+1, group=None)
        
        Cij = torch.matmul(Xij, W0.mT)
        send_B_1.wait()
        recv_B_1.wait()
        
    elif mod == 1:
        XW = torch.matmul(Xij, W0.mT)
        recv_B_0 = torch.distributed.irecv(bufferXW, src=rank-1, group=None)
        send_B_0 = torch.distributed.isend(XW, dst=rank-1, group=None)
        
        Cij = torch.matmul(Xij, W1.mT)

        send_B_0.wait()
        recv_B_0.wait()

    if mod == 0:
        Cij = Cij + bufferXW
    else:
        Cij = Cij + bufferXW
    
    return Cij


def XTW_twoway(Xij, Wij, device, rank, model_group=None):
    """
    Perform a two-way asynchronous tensor broadcast and matrix multiplication.
    Parameters
    ----------
    Xij : torch.Tensor
        The input tensor X.
    Wij : torch.Tensor
        The input tensor W.
    device : torch.device
        The device on which to perform the operations.
    rank : int
        The rank of the current process in the distributed group.
    model_group : torch.distributed.ProcessGroup, optional
        The process group for communication (default is None).
    Returns
    -------
    torch.Tensor
        The resulting tensor after performing the matrix multiplications and concatenation.
    """

    
    mod = rank % 2

    bufferX = get_preallocated_buffer(Xij.shape, torch.float32, device, rank)
    
    if mod == 0: 
        send_B_1 =torch.distributed.isend(Xij, dst=rank+1, group=None)
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
