from mpi4py import MPI
import numpy as np
import torch
import torch.distributed as dist

def AB(Aij, Bij, rank, world_size, device):
    
    #assert(Aij.shape[1] == Bij.shape[0])
    #m, n, p = Aij.shape[0], Aij.shape[1], Bij.shape[1] 
       
    #Cij = torch.zeros(m, p).to(device)
    
    sqrtp = int(np.sqrt(world_size))
    #subset_size_m, subset_size_n, subset_size_p = m//sqrtp, n//sqrtp, p//sqrtp
    
    # Empty allocation of A_k per process
    A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)

    bufferA = torch.zeros_like(Aij).contiguous()
    bufferB = torch.zeros_like(Bij).contiguous()

    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        # Manual send of A
        A_k = distribute_A(Aij, k, rank, bufferA, device)

        B_k = distribute_B(Bij, k, rank, bufferB, device)
        
        ## Perform local multiplication and accumulate
        Cij = Cij + torch.matmul(A_k, B_k)
        
    return Cij


def ABT_vert(Aij, Bij):
    
    Cij = Aij @ Bij.mT

    torch.distributed.all_reduce(Cij) 
        
    return Cij

def AB_vert(Aij, Bij):
    Cij = Aij @ Bij

    tensor_out = torch.zeros(4, *Cij.shape).to(Aij.device)
    torch.distributed.all_gather_into_tensor(tensor_out, Cij) # [4, n_batch, patches, 36]
    tensor_out = tensor_out.permute(1, 2, 0, 3)
    tensor_out = tensor_out.reshape(Aij.shape[0], -1, Bij.shape[-1])
    
    return tensor_out


def ATB_vert(Aij, Bij):
    Cij = Aij.mT @ Bij
    
    tensor_out = torch.zeros(4, *Cij.shape).to(Aij.device)
    torch.distributed.all_gather_into_tensor(tensor_out, Cij) # [4, n_batch, patches, 36]
    tensor_out = tensor_out.permute(1, 2, 0, 3)
    tensor_out = tensor_out.reshape(Cij.shape[0], -1, Bij.shape[-1])
    
    return tensor_out
    
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
        
def distribute_A(Aij, k, rank, bufferA, device):
    A_k = Aij.clone().detach()

    if k == 0:
        if rank == 0:
            dist.isend(Aij, dst=1)
        elif rank == 1:
            req2 = dist.irecv(bufferA, src=0)
            req2.wait()
            A_k = bufferA.to(device)
        elif rank == 2:
            dist.isend(Aij, dst=3)
        else:
            req2 = dist.irecv(bufferA, src=2)
            req2.wait()
            A_k = bufferA.to(device)
    elif k == 1:
        if rank == 0:
            req2 = dist.irecv(bufferA, src=1)
            req2.wait()
            A_k = bufferA.to(device)
        elif rank == 1:
            dist.isend(Aij, dst=0)
        elif rank == 2:
            req2 = dist.irecv(bufferA, src=3)
            req2.wait()
            A_k = bufferA.to(device)
        else:
            dist.isend(Aij, dst=2)
    return A_k

def distribute_B(Bij, k, rank, bufferB, device):
    Bij = Bij.contiguous()
    B_k = Bij.clone().detach()
    if k == 0:
        if rank == 0:
            dist.isend(Bij, dst=1)
        elif rank == 1:
            req2 = dist.irecv(bufferB, src=0)
            req2.wait()
            B_k = bufferB.to(device)
        elif rank == 2:
            dist.isend(Bij, dst=3)
        else:
            req2 = dist.irecv(bufferB, src=2)
            req2.wait()
            B_k = bufferB.to(device)
    elif k == 1:
        if rank == 0:
            req2 = dist.irecv(bufferB, src=1)
            req2.wait()
            B_k = bufferB.to(device)
        elif rank == 1:
            dist.isend(Bij, dst=0)
        elif rank == 2:
            req2 = dist.irecv(bufferB, src=3)
            req2.wait()
            B_k = bufferB.to(device)
        else:
            dist.isend(Bij, dst=2)
    return B_k

def matrix_transpose(X, rank, device):
    X_T = X.clone().detach()
    buffer = torch.zeros_like(X.mT).contiguous()
    if rank == 0 or rank == 3:
        X_T = X.mT
    elif rank == 1:
        req_send = dist.isend(X.mT.contiguous(), dst=2)
        req_recv = dist.irecv(buffer, src=2)
        req_recv.wait()
        req_send.wait()
        X_T = buffer.to(device)
    elif rank == 2:
        req_recv = dist.irecv(buffer, src=1)
        req_send = dist.isend(X.mT.contiguous(), dst=1)
        req_recv.wait()
        req_send.wait()
        X_T = buffer.to(device)
    
    return X_T


def ABT(Aij, Bij, rank, world_size, device):
    sqrtp = int(np.sqrt(world_size))
    
    rank_dict_transpose = {}
    rank_dict_norm = {}
    """
    rank_dict_transpose = {
        0: [0, 2], # for rank % k == 0
        2: [1, 3], # for rank % k == 0
        1: [0, 2], # for rank % k == 1
        3: [1, 3]  # for rank % k == 1        
    }
    """
    for r in range(world_size):
        rank_dict_transpose[transpose(sqrtp, r)] = [k * sqrtp + int(r % np.sqrt(world_size)) for k in range(sqrtp)]
        rank_dict_norm[r] = [k * sqrtp + int(r % np.sqrt(world_size)) for k in range(sqrtp)]
    
    A_k = torch.zeros_like(Aij).detach().to(device)
    
    Cij = None 
    
    bufferA = torch.zeros_like(Aij).contiguous()
    bufferB = torch.zeros_like(Bij).contiguous()

    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        # Manual send of A
        root_ranks = np.arange(sqrtp)*sqrtp + k
        A_k = distribute_A(Aij, k, rank, bufferA, device)
        
        B_k = Bij.clone().detach()
        
        root_ranks = np.arange(sqrtp)*sqrtp + k
        ## sending
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.isend(Bij.contiguous(), dst=target)
                    
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        req = dist.irecv(bufferB, src=root_rank)
                        req.wait()
                        B_k = bufferB.to(A_k.device) 
        
        if Cij is None:
            Cij = A_k @ B_k.mT
        else:
            Cij = Cij + A_k @ B_k.mT
    

    return Cij

def ATB(Aij, Bij, rank, world_size, device):
    
    sqrtp = int(np.sqrt(world_size))
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))


    rank_dict_transpose = {}
    """
    rank_dict_transpose = {
        0: [0, 1], # for rank % k == 0
        2: [2, 3], # for rank % k == 0
        1: [2, 3], # for rank % k == 1
        3: [2, 3]  # for rank % k == 1        
    }
    """
    for r in range(world_size):
        rank_dict_transpose[transpose(sqrtp, r)] = [int(r // np.sqrt(world_size) * sqrtp + k) for k in range(sqrtp)]
    #if rank == 0:
    #    print(rank_dict_transpose)

    A_k = torch.zeros_like(Aij)
    B_k = torch.zeros_like(Bij)

    
    # TODO matmul dimensions currently fixed. need to generalize?
    # TODO : VERY INEFFICIENT, hacky way of getting correct shape
    Cij = torch.zeros((A_k.mT @ B_k).shape).to(B_k.device)
    #Cij = torch.zeros(Aij.shape[:len_Aij-2], Bij.shape[0]).to(device).to(B_k.device)


    buffer  = torch.zeros_like(Aij)
    bufferB = torch.zeros_like(Bij)

    
    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        A_k = Aij.clone()

        # for k = 0
        # "root" ranks: [0, 1]
        # for k = 1, root_ranks [2, 3]
        root_ranks = np.arange(sqrtp*k, sqrtp*(k+1))

        # sending
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.isend(Aij, dst=target)    
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        req = dist.irecv(buffer, src=root_rank)
                        req.wait()
                        A_k = buffer.to(device)


        B_k = distribute_B(Bij, k, rank, bufferB, device)
        Cij = Cij + A_k.mT @ B_k

    return Cij
