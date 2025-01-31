from mpi4py import MPI
import numpy as np
import torch
import torch.distributed as dist

def AB(Aij, Bij, rank, n_gpus_per_node, device):
    
    sqrtp = int(np.sqrt(n_gpus_per_node))
        
    # Empty allocation of A_k per process
    A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)

    for k in range(sqrtp):

        bufferA = torch.zeros_like(Aij).contiguous()
        bufferB = torch.zeros_like(Bij).contiguous()

        # Broadcast submatrix of A along the row
        # Manual send of A
        A_k = distribute_A(Aij, k, rank, bufferA)
        B_k = distribute_B(Bij, k, rank, bufferB)
        
        ## Perform local multiplication and accumulate
        Cij = Cij + torch.matmul(A_k, B_k)
        
    return Cij



def transpose(n, rank):
    C = torch.arange(n*n).reshape(n, n).mT.reshape(-1)
    transpose_index = int(C[rank])
    return transpose_index
    

def gather(Cij, comm, world_size):
    rank = comm.Get_rank()
    p = int(np.sqrt(world_size))
    
    C_collective = comm.gather(Cij, root=0)

    if rank == 0:
        C_collective = torch.stack(C_collective)
        C_collective = C_collective.reshape(p, p, *C_collective.shape[1:])
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(Cij.shape[0]*p, Cij.shape[1]*p)

    return C_collective
        
def distribute_A(Aij, k, rank, bufferA):
    if k == 0:
        if rank % 4 == 0:
            dist.isend(Aij, dst=rank+1)
            A_k = Aij
        elif rank % 4 == 1:
            req2 = dist.irecv(bufferA, src=rank-1)
            req2.wait()
            A_k = bufferA
        elif rank % 4 == 2:
            dist.isend(Aij, dst=rank+1)
            A_k = Aij
        else:
            req2 = dist.irecv(bufferA, src=rank-1)
            req2.wait()
            A_k = bufferA
    elif k == 1:
        if rank % 4 == 0:
            req2 = dist.irecv(bufferA, src=rank+1)
            req2.wait()
            A_k = bufferA
        elif rank % 4 == 1:
            dist.isend(Aij, dst=rank-1)
            A_k = Aij
        elif rank % 4 == 2:
            req2 = dist.irecv(bufferA, src=rank+1)
            req2.wait()
            A_k = bufferA
        else:
            dist.isend(Aij, dst=rank-1)
            A_k = Aij
        
    return A_k

def distribute_B(Bij, k, rank, bufferB):
    if k == 0:
        if rank % 4 == 0:
            dist.isend(Bij, dst=rank+2)
            B_k = Bij
        elif rank % 4 == 1:
            dist.isend(Bij, dst=rank+2)
            B_k = Bij
        elif rank % 4 == 2:
            req2 = dist.irecv(bufferB, src=rank-2)
            req2.wait()
            B_k = bufferB
        else:
            req2 = dist.irecv(bufferB, src=rank-2)
            req2.wait()
            B_k = bufferB
    elif k == 1:
        if rank % 4 == 0:
            req2 = dist.irecv(bufferB, src=rank+2)
            req2.wait()
            B_k = bufferB
        elif rank % 4 == 1:
            req2 = dist.irecv(bufferB, src=rank+2)
            req2.wait()
            B_k = bufferB
        elif rank % 4 == 2:
            dist.isend(Bij, dst=rank-2)
            B_k = Bij
        else:
            dist.isend(Bij, dst=rank-2)
            B_k = Bij
    return B_k

def matrix_transpose(X, rank, device):
    if rank % 4 == 1:
        buffer = torch.zeros(X.shape).to(device)
        req_send = dist.isend(X, dst=rank+1)
        req_recv = dist.irecv(buffer, src=rank+1)
        req_recv.wait()
        req_send.wait()
        X_T = buffer
    elif rank % 4 == 2:
        buffer = torch.zeros(X.shape).to(device)
        req_recv = dist.irecv(buffer, src=rank-1)
        req_send = dist.isend(X, dst=rank-1)
        req_recv.wait()
        req_send.wait()
        X_T = buffer
    else:
        X_T = X
    
    return X_T.mT


def ABT(Aij, Bij, rank, n_gpus_per_node, device):
    sqrtp = int(np.sqrt(n_gpus_per_node))
    
    rank_dict_transpose = {}
    
    """
    rank_dict_transpose = {
        0: [0, 2], # for rank % k == 0
        2: [1, 3], # for rank % k == 0
        1: [0, 2], # for rank % k == 1
        3: [1, 3]  # for rank % k == 1        
    }
    """
    node_rank = rank // n_gpus_per_node

    for r in range(4):
        rank_dict_transpose[transpose(sqrtp, r) + node_rank * n_gpus_per_node] = [k * sqrtp + int(r % np.sqrt(n_gpus_per_node)) + node_rank * n_gpus_per_node for k in range(sqrtp)]
    
    Cij = None 
    
    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        # Manual send of A
        root_ranks = np.arange(sqrtp)*sqrtp + k + node_rank * n_gpus_per_node
        bufferA = torch.zeros(Aij.shape).to(device)
        A_k = distribute_A(Aij, k, rank, bufferA)
        
        B_k = Bij.clone().detach()
        
        root_ranks = np.arange(sqrtp)*sqrtp + k + node_rank * n_gpus_per_node
        
        ## sending
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.isend(Bij, dst=target)
                    
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        bufferB = torch.zeros(Bij.shape).to(device)
                        req = dist.irecv(bufferB, src=root_rank)
                        req.wait()
                        B_k = bufferB
        
        if Cij is None:
            Cij = A_k @ B_k.mT
        else:
            Cij = Cij + A_k @ B_k.mT

    return Cij

def ATB(Aij, Bij, rank, n_gpus_per_node, device):
    
    sqrtp = int(np.sqrt(n_gpus_per_node))
    
    rank_dict_transpose = {}
    """
    rank_dict_transpose = {
        0: [0, 1], # for rank % k == 0
        2: [2, 3], # for rank % k == 0
        1: [2, 3], # for rank % k == 1
        3: [2, 3]  # for rank % k == 1        
    }
    """
    node_rank = rank // n_gpus_per_node

    for r in range(n_gpus_per_node):
        rank_dict_transpose[transpose(sqrtp, r) + node_rank * n_gpus_per_node] = [int(r // np.sqrt(n_gpus_per_node) * sqrtp + k) + node_rank * n_gpus_per_node for k in range(sqrtp)]
    #if rank == 0:
    #    print(rank_dict_transpose)

    A_k = torch.zeros(Aij.shape).to(device)
    B_k = torch.zeros(Bij.shape).to(device)

    Cij = None 
    
    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        # for k = 0
        # "root" ranks: [0, 1]
        # for k = 1, root_ranks [2, 3]
        root_ranks = np.arange(sqrtp*k, sqrtp*(k+1)) + node_rank * n_gpus_per_node
        bufferB = torch.zeros(Bij.shape).to(device)
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.isend(Aij, dst=target)    
                    A_k = Aij
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        buffer  = torch.zeros(Aij.shape).to(device)
                        req = dist.irecv(buffer, src=root_rank)
                        req.wait()
                        A_k = buffer


        B_k = distribute_B(Bij, k, rank, bufferB)

        if Cij is None:
            Cij = A_k.mT @ B_k
        else:
            Cij = Cij + A_k.mT @ B_k

    return Cij