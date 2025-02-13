#from mpi4py import MPI
import numpy as np
import torch
import torch.distributed as dist
import time

def AB(Aij, Bij, rank, n_gpus_per_node, device):
    
    sqrtp = int(np.sqrt(n_gpus_per_node))
        
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)

    for k in range(sqrtp):

        bufferA = torch.zeros_like(Aij).to(torch.float32).contiguous()
        bufferB = torch.zeros_like(Bij).to(torch.float32).contiguous()

        # Broadcast submatrix of A along the row
        # Manual send of A
        A_k = distribute_A(Aij, k, rank, bufferA)
        B_k = distribute_B(Bij, k, rank, bufferB)
        
        ## Perform local multiplication and accumulate
        if Cij is None:
            Cij = torch.matmul(A_k, B_k)
        else:
            Cij = Cij + torch.matmul(A_k, B_k)
        
    return Cij

def AB_custom(Aij, Bij, rank, n_gpus_per_node, device):
    
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    bufferA = get_preallocated_buffer(Aij.shape, Aij.dtype, device)
    bufferB = get_preallocated_buffer(Bij.shape, Bij.dtype, device)
    
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_1 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
        reqs = dist.batch_isend_irecv([send_A_1, send_B_2, recv_A_1, recv_B_2])
        Cij = torch.matmul(Aij, Bij)
    elif mod == 1:
        send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_3 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_3 = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
        reqs = dist.batch_isend_irecv([send_A_0, send_B_3, recv_A_0, recv_B_3])
    elif mod == 2:
        send_A_3 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_3 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB, peer=rank-2)
        reqs = dist.batch_isend_irecv([send_A_3, send_B_0, recv_A_3, recv_B_0])
    else:
        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB, peer=rank-2)
        reqs = dist.batch_isend_irecv([send_A_2, send_B_1, recv_A_2, recv_B_1])
        Cij = torch.matmul(Aij, Bij)

    for req in reqs:
        req.wait()

    if mod in [0, 3]:
        Cij = Cij + torch.matmul(bufferA, bufferB)        
    else:
        Cij = torch.matmul(bufferA, Bij)  + torch.matmul(Aij, bufferB)      
    
    return Cij

def AB_custom_isend(Aij, Bij, rank, n_gpus_per_node, device):
    
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    bufferA = torch.zeros_like(Aij).to(torch.float32).contiguous()
    bufferB = torch.zeros_like(Bij).to(torch.float32).contiguous()

    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_1 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
        reqs = dist.batch_isend_irecv([send_A_1, send_B_2, recv_A_1, recv_B_2])
        Cij = torch.matmul(Aij, Bij)
    elif mod == 1:
        send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        recv_A_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)

        send_B_3 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_B_3 = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
        reqsA = dist.batch_isend_irecv([send_A_0, send_B_3, recv_A_0])
        reqsB = dist.batch_isend_irecv([recv_B_3])

    elif mod == 2:
        send_A_3 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        recv_A_3 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB, peer=rank-2)

        reqsA = dist.batch_isend_irecv([send_A_3, send_B_0, recv_A_3])
        reqsB = dist.batch_isend_irecv([recv_B_0])

    else:
        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB, peer=rank-2)
        reqs = dist.batch_isend_irecv([send_A_2, send_B_1, recv_A_2, recv_B_1])
        Cij = torch.matmul(Aij, Bij)

    if mod in [0, 3]:
        for req in reqs:
            req.wait()

        Cij = Cij + torch.matmul(bufferA, bufferB)        
    else:
        for req in reqsA:
            req.wait()
        Cij = torch.matmul(bufferA, Bij)

        for req in reqsB:
            req.wait()
        Cij = Cij + torch.matmul(Aij, bufferB)      
    
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
        if rank % 4 in [0,2]:
            send_op = dist.P2POp(dist.isend, Aij, peer=rank+1)
            reqs = dist.batch_isend_irecv([send_op])
            bufferA = Aij
        else:
            recv_op = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
            reqs = dist.batch_isend_irecv([recv_op])
    else:
        if rank % 4 in [1,3]:
            send_op = dist.P2POp(dist.isend, Aij, peer=rank-1)
            reqs = dist.batch_isend_irecv([send_op])
            bufferA = Aij
        else:
            recv_op = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
            reqs = dist.batch_isend_irecv([recv_op])
    for req in reqs:
        req.wait()
    A_k = bufferA
        
    return A_k

def distribute_B(Bij, k, rank, bufferB):
    #if k == 0:
    #    if rank % 4 == 0:
    #        dist.send(Bij, dst=rank+2)
    #        B_k = Bij
    #    elif rank % 4 == 1:
    #        dist.send(Bij, dst=rank+2)
    #        B_k = Bij
    #    elif rank % 4 == 2:
    #        req2 = dist.recv(bufferB, src=rank-2)
    #        #req2.wait()
    #        B_k = bufferB
    #    else:
    #        req2 = dist.recv(bufferB, src=rank-2)
#   #         req2.wait()
    #        B_k = bufferB
    #elif k == 1:
    #    if rank % 4 == 0:
    #        req2 = dist.recv(bufferB, src=rank+2)
    #        #req2.wait()
    #        B_k = bufferB
    #    elif rank % 4 == 1:
    #        req2 = dist.recv(bufferB, src=rank+2)
    #        #req2.wait()
    #        B_k = bufferB
    #    elif rank % 4 == 2:
    #        dist.send(Bij, dst=rank-2)
    #        B_k = Bij
    #    else:
    #        dist.send(Bij, dst=rank-2)
    #        B_k = Bij
    if k == 0:
        if rank % 4 in [0,1]:
            send_op = dist.P2POp(dist.isend, Bij, peer=rank+2)
            reqs = dist.batch_isend_irecv([send_op])
            bufferB = Bij
        else:
            recv_op = dist.P2POp(dist.irecv, bufferB, peer=rank-2)
            reqs = dist.batch_isend_irecv([recv_op])
    else:
        if rank % 4 in [2,3]:
            send_op = dist.P2POp(dist.isend, Bij, peer=rank-2)
            reqs = dist.batch_isend_irecv([send_op])
            bufferB = Bij
        else:
            recv_op = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
            reqs = dist.batch_isend_irecv([recv_op])
    for req in reqs:
        req.wait()
    B_k = bufferB
    return B_k

def matrix_transpose(X, rank, device):
    if rank % 4 == 1:
        #buffer = torch.zeros(X.shape).to(torch.float32).to(device)
        buffer = get_preallocated_buffer(X.shape, X.dtype, device)
        req_send = dist.send(X, dst=rank+1)
        req_recv = dist.recv(buffer, src=rank+1)
        #req_recv.wait()
        #req_send.wait()
        X_T = buffer
    elif rank % 4 == 2:
        #buffer = torch.zeros(X.shape).to(torch.float32).to(device)
        buffer = get_preallocated_buffer(X.shape, X.dtype, device)
        req_recv = dist.recv(buffer, src=rank-1)
        req_send = dist.send(X, dst=rank-1)
        #req_recv.wait()
        #req_send.wait()
        X_T = buffer
    else:
        X_T = X
    
    return X_T.mT

def matrix_transpose_twoway(X, rank, device):
    mod = rank % 2

    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 2
    X0, X1 = torch.split(X, X.shape[-2]//2, dim=-2)
        
    bufferX = get_preallocated_buffer(X1.shape, X.dtype, device) 
    
    
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_B_1 = dist.P2POp(dist.isend, X1, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferX, peer=rank+1)
        reqs = dist.batch_isend_irecv([send_B_1, recv_B_1])
    elif mod == 1:
           
        send_B_0 = dist.P2POp(dist.isend, X0, peer=rank-1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferX, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_B_0, recv_B_0])
    
    for req in reqs:
        req.wait()
    if mod == 0:
        XT = torch.concat([X0.mT, bufferX.mT], dim=-2)
    elif mod == 1:
        XT = torch.concat([bufferX.mT, X1.mT], dim=-2)
    return XT


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
        bufferA = torch.zeros(Aij.shape).to(torch.float32).to(device)
        A_k = distribute_A(Aij, k, rank, bufferA)
        
        B_k = Bij.clone().detach()
        
        root_ranks = np.arange(sqrtp)*sqrtp + k + node_rank * n_gpus_per_node
        
        ## sending
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.send(Bij, dst=target)
                    
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        bufferB = torch.zeros(Bij.shape).to(torch.float32).to(device)
                        req = dist.recv(bufferB, src=root_rank)
                        #req.wait()
                        B_k = bufferB
        
        if Cij is None:
            Cij = A_k @ B_k.mT
        else:
            Cij = Cij + A_k @ B_k.mT

    return Cij


def ABT_custom(Aij, Bij, rank, n_gpus_per_node, device):
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    
    if mod == 0: 
        bufferA = torch.zeros(Aij.shape).to(torch.float32).to(device)
        bufferB = torch.zeros(Bij.shape).to(torch.float32).to(device)
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_1 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB, peer=rank+1)
        reqs = dist.batch_isend_irecv([send_A_1, send_B_2, recv_A_1, recv_B_1])
        Cij = torch.matmul(Aij, Bij.mT)
    elif mod == 1:
        bufferA = torch.zeros(Aij.shape).to(torch.float32).to(device)
        bufferB2 = torch.zeros(Bij.shape).to(torch.float32).to(device)
        bufferB3 = torch.zeros(Bij.shape).to(torch.float32).to(device)

        send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+1)

        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)
        recv_A_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB2, peer=rank+1)     
        reqs = dist.batch_isend_irecv([send_A_0, send_B_0, send_B_2, recv_B_3, recv_A_0, recv_B_2])
    elif mod == 2:
        bufferA  = torch.zeros(Aij.shape).to(torch.float32).to(device)
        bufferB0 = torch.zeros(Bij.shape).to(torch.float32).to(device)
        bufferB1 = torch.zeros(Bij.shape).to(torch.float32).to(device)
        
        send_A_3 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_3 = dist.P2POp(dist.isend, Bij, peer=rank+1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-1)

        recv_A_3 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB1, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_A_3, send_B_3, send_B_1, recv_B_0, recv_A_3, recv_B_1])
    else:
        bufferA = torch.zeros(Aij.shape).to(torch.float32).to(device)
        bufferB = torch.zeros(Bij.shape).to(torch.float32).to(device)
        
        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_A_2, send_B_1, recv_A_2, recv_B_2])
        Cij = torch.matmul(Aij, Bij.mT)
        
    if mod in [0, 3]:
        for req in reqs:
            req.wait()
        Cij = Cij + torch.matmul(bufferA, bufferB.mT)        
    else:
        for req in reqs:
            req.wait()
        
        if mod == 1:
            Cij = torch.matmul(Aij, bufferB3.mT) + torch.matmul(bufferA, bufferB2.mT)
        else:
            Cij = torch.matmul(Aij, bufferB0.mT) + torch.matmul(bufferA, bufferB1.mT)
    
    return Cij


def ABT_custom_precompute(Aij, Bij, rank, n_gpus_per_node, device):
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    
    if mod == 0: 
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((*Aij.shape[:-1], Bij.shape[0]), Aij.dtype, device)
        else:
            bufferAB = get_preallocated_buffer((Bij.shape[0], Aij.shape[0], Bij.shape[-2]), Aij.dtype, device)
        
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_AB_1 = dist.P2POp(dist.irecv, bufferAB, peer=rank+1)
        
        reqs0 = dist.batch_isend_irecv([send_A_1, send_B_2])
        Cij = torch.matmul(Aij, Bij.mT)
        reqs1 = dist.batch_isend_irecv([recv_AB_1])
        #print(f'buffer shape {bufferAB.shape}, Cij shape {Cij.shape}')
        
    elif mod == 1:

        bufferA  = get_preallocated_buffer(Aij.shape, Aij.dtype, device)
        bufferB2 = get_preallocated_buffer(Bij.shape, Bij.dtype, device) 
        bufferB3 = get_preallocated_buffer(Bij.shape, Bij.dtype, device) 

        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+1)

        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)
        recv_A_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB2, peer=rank+1)     
        reqs0 = dist.batch_isend_irecv([send_B_2, recv_B_3, recv_A_0, recv_B_2])

        AB = torch.matmul(Aij, Bij.mT)
        
        send_AB_0 = dist.P2POp(dist.isend, AB, peer=rank-1)
        reqs1 = dist.batch_isend_irecv([send_AB_0])
        
                  
    elif mod == 2:
        bufferA  = get_preallocated_buffer(Aij.shape, Aij.dtype, device)
        bufferB0 = get_preallocated_buffer(Bij.shape, Bij.dtype, device) 
        bufferB1 = get_preallocated_buffer(Bij.shape, Bij.dtype, device) 
        
        
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-1)

        recv_A_3 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB1, peer=rank-1)
        reqs0 = dist.batch_isend_irecv([send_B_1, recv_B_0, recv_A_3, recv_B_1])
        AB = torch.matmul(Aij, Bij.mT)
        send_AB_3 = dist.P2POp(dist.isend, AB, peer=rank+1)
        reqs1 = dist.batch_isend_irecv([send_AB_3])

    else:
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((*Aij.shape[:-1], Bij.shape[0]), Aij.dtype, device)
        else:
            bufferAB = get_preallocated_buffer((Bij.shape[0], Aij.shape[0], Bij.shape[-2]), Aij.dtype, device)
        
        send_A_2  = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1  = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_AB_2 = dist.P2POp(dist.irecv, bufferAB, peer=rank-1)
        reqs0 = dist.batch_isend_irecv([send_A_2, send_B_1])
        Cij   = torch.matmul(Aij, Bij.mT)
        reqs1 = dist.batch_isend_irecv([recv_AB_2])
        
    for req in reqs0:
        req.wait()
    for req in reqs1:
        req.wait()

    if mod  in [0, 3]:
        Cij = Cij + bufferAB
    elif mod == 1:
        Cij = torch.matmul(Aij, bufferB3.mT) + torch.matmul(bufferA, bufferB2.mT)
    else:
        Cij = torch.matmul(Aij, bufferB0.mT) + torch.matmul(bufferA, bufferB1.mT)
    
    return Cij

def ABT_custom_isend(Aij, Bij, rank, n_gpus_per_node, device):
    Cij = None 
    mod = rank % 4
    #print(f'rank {rank} has made it beginning')
    if mod == 0: 
        bufferA1 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB1 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_1 = dist.P2POp(dist.irecv, bufferA1, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB1, peer=rank+1)
        
        reqs0 = dist.batch_isend_irecv([send_B_2])
        
        Cij = torch.matmul(Aij, Bij.mT)
        for req in reqs0:
            req.wait()
        #print(f'{rank} made it')
        reqs1 = dist.batch_isend_irecv([send_A_1, recv_A_1, recv_B_1])
        for req in reqs1:
            req.wait()
        
    elif mod == 1:
        bufferA0 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB2 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        bufferB3 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)

        send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+1)

        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)
        recv_A_0 = dist.P2POp(dist.irecv, bufferA0, peer=rank-1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB2, peer=rank+1)     
 
        reqs0 = dist.batch_isend_irecv([recv_B_3])
        for req in reqs0:
            req.wait()
        #print(f'{rank} made it')
        Cij = torch.matmul(Aij, bufferB3.mT) 
        reqs1 = dist.batch_isend_irecv([send_A_0, send_B_0, send_B_2, recv_A_0, recv_B_2])
        for req in reqs1:
            req.wait()
        
    elif mod == 2:
        bufferA3 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB0 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        bufferB1 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        
        send_A_3 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_3 = dist.P2POp(dist.isend, Bij, peer=rank+1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-1)

        recv_A_3 = dist.P2POp(dist.irecv, bufferA3, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB1, peer=rank-1)
  
        reqs0 = dist.batch_isend_irecv([recv_B_0])

        for req in reqs0:
            req.wait()
        #print(f'{rank} made it')
        Cij = torch.matmul(Aij, bufferB0.mT) 

        reqs1 = dist.batch_isend_irecv([send_A_3, send_B_3, send_B_1, recv_A_3, recv_B_1])
        for req in reqs1:
            req.wait()
    else:
        bufferA2 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB2 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        
        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA2, peer=rank-1)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB2, peer=rank-1)
   
        reqs0 = dist.batch_isend_irecv([send_B_1])        
        Cij = torch.matmul(Aij, Bij.mT)
        for req in reqs0:
            req.wait()
        #print(f'{rank} made it')
        reqs1 = dist.batch_isend_irecv([send_A_2, recv_A_2, recv_B_2])
        for req in reqs1:
            req.wait()

    if mod == 0:
        Cij = Cij + torch.matmul(bufferA1, bufferB1.mT)        
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferA0, bufferB2.mT)
    elif mod == 2:
        Cij = Cij + torch.matmul(bufferA3, bufferB1.mT)
    else:
        Cij = Cij +  torch.matmul(bufferA2, bufferB2.mT)        
    #print(f'rank {rank} has made it to end')
    
    return Cij



def ATB_custom(Aij, Bij, rank, n_gpus_per_node, device):
    
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    
    if mod == 0: 
        bufferA = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2 = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA, peer=rank+2)
        recv_B_2 = dist.P2POp(dist.irecv, bufferB, peer=rank+2)
        reqs = dist.batch_isend_irecv([send_A_1, send_B_2, recv_A_2, recv_B_2])
        Cij = torch.matmul(Aij.mT, Bij)
    elif mod == 1:
        bufferA0 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferA2 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB3 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)

        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_A_3 = dist.P2POp(dist.isend, Aij, peer=rank+2)
        send_B_3 = dist.P2POp(dist.isend, Bij, peer=rank+2)

        recv_A_0 = dist.P2POp(dist.irecv, bufferA0, peer=rank-1)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA2, peer=rank+1)
        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)
        reqs = dist.batch_isend_irecv([send_A_2, send_A_3, send_B_3, recv_A_0, recv_A_2, recv_B_3])
    elif mod == 2:
        bufferA1 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferA3 = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB0 = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        
        send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-2)
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-2)

        recv_A_1 = dist.P2POp(dist.irecv, bufferA1, peer=rank-1)
        recv_A_3 = dist.P2POp(dist.irecv, bufferA3, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        reqs = dist.batch_isend_irecv([send_A_0, send_A_1, send_B_0, recv_A_3, recv_A_1, recv_B_0])
    else:
        bufferA = torch.zeros(Aij.shape).to(Aij.dtype).to(device)
        bufferB = torch.zeros(Bij.shape).to(Bij.dtype).to(device)
        
        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1 = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_A_1 = dist.P2POp(dist.irecv, bufferA, peer=rank-2)
        recv_B_1 = dist.P2POp(dist.irecv, bufferB, peer=rank-2)
        reqs = dist.batch_isend_irecv([send_A_2, send_B_1, recv_A_1, recv_B_1])
        Cij = torch.matmul(Aij.mT, Bij)
        
    for req in reqs:
        req.wait()

    if mod in [0, 3]:
        Cij = Cij + torch.matmul(bufferA.mT, bufferB)
    elif mod == 1:
        Cij = torch.matmul(bufferA0.mT, Bij) + torch.matmul(bufferA2.mT, bufferB3) 
    else:
        Cij = torch.matmul(bufferA3.mT, Bij) + torch.matmul(bufferA1.mT, bufferB0) 

    return Cij

_prealloc_buffer = None
_prealloc_numel = 0

def get_preallocated_buffer(shape, dtype, device):
    global _prealloc_buffer, _prealloc_numel

    needed = 1
    _prealloc_numel = 1
    for dim in shape: 
        needed *= dim

    # Reallocate if buffer is None or too small
    if _prealloc_buffer is None or needed > _prealloc_numel:
        _prealloc_buffer = torch.zeros(shape, dtype=dtype, device=device)
        _prealloc_numel = needed
    else:
        _prealloc_buffer.zero_()
    return _prealloc_buffer.view(*shape)


def ATB_custom_precompute(Aij, Bij, rank, n_gpus_per_node, device):
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    
    if mod == 0: 
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((Aij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        else:
            bufferAB = get_preallocated_buffer(Bij.shape[0], Aij.shape[-1], Bij.shape[-1]).to(Aij.dtype)
                    

        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1  = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2  = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_AB_2 = dist.P2POp(dist.irecv, bufferAB, peer=rank+2)
        
        reqs0 = dist.batch_isend_irecv([send_A_1, send_B_2])
        Cij = torch.matmul(Aij.mT, Bij)
        reqs1 = dist.batch_isend_irecv([recv_AB_2])

    elif mod == 1:
        bufferA0 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferA2 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferB3 = get_preallocated_buffer((Bij.shape), Bij.dtype, device)

        
        send_A_2  = dist.P2POp(dist.isend, Aij, peer=rank+1)
        
        recv_A_0 = dist.P2POp(dist.irecv, bufferA0, peer=rank-1)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA2, peer=rank+1)
        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)
        reqs0 = dist.batch_isend_irecv([send_A_2, recv_A_0, recv_A_2, recv_B_3])
    
        AB = torch.matmul(Aij.mT, Bij)
        send_AB_3 = dist.P2POp(dist.isend, AB, peer=rank+2)
        reqs1 = dist.batch_isend_irecv([send_AB_3])

    elif mod == 2:
        bufferA1 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferA3 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferB0 = get_preallocated_buffer((Bij.shape), Bij.dtype, device)
        
        send_A_1  = dist.P2POp(dist.isend, Aij, peer=rank-1)
        
        recv_A_1 = dist.P2POp(dist.irecv, bufferA1, peer=rank-1)
        recv_A_3 = dist.P2POp(dist.irecv, bufferA3, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        reqs0 = dist.batch_isend_irecv([send_A_1, recv_A_3, recv_A_1, recv_B_0])
        AB = torch.matmul(Aij.mT, Bij)
        send_AB_0 = dist.P2POp(dist.isend, AB, peer=rank-2)
        reqs1 = dist.batch_isend_irecv([send_AB_0])
    else:
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((Aij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        else:
            bufferAB = get_preallocated_buffer((Bij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        
        send_A_2  = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1  = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_AB_1 = dist.P2POp(dist.irecv, bufferAB, peer=rank-2)

        reqs0 = dist.batch_isend_irecv([send_A_2, send_B_1])
        Cij = torch.matmul(Aij.mT, Bij)
        reqs1 = dist.batch_isend_irecv([recv_AB_1])

    for req in reqs0:
        req.wait()
    for req in reqs1:
        req.wait()
   
            
    if mod in [0, 3]:
        Cij = Cij + bufferAB
    elif mod == 1:
        Cij = torch.matmul(bufferA0.mT, Bij) + torch.matmul(bufferA2.mT, bufferB3) 
    else:
        Cij = torch.matmul(bufferA3.mT, Bij) + torch.matmul(bufferA1.mT, bufferB0) 
    

    #if rank == 0:
    #    #print(f'Rank {rank}: {round(message_to_buffer - beginning_time, 4)} s to allocate buffer; {round(wait_end - wait_start, 6)} s to receive;  {round(mat_end - wait_end, 6)} s multiply {round(mat_end - beginning_time, 5)} s total')
    #    print(f'Rank {rank}, {round(mat_end - beginning_time, 5)} s total')
    #elif rank == 1:
    #    print(f'Rank {rank}, {round(mat_end - beginning_time, 5)} s total')
    
    return Cij


def ATB_custom_isend(Aij, Bij, rank, n_gpus_per_node, device):
    
    # Empty allocation of A_k per process
    # A_k = torch.zeros_like(Aij).detach().to(device)
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 4
    
    if mod == 0: 
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((Aij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        else:
            bufferAB = torch.zeros(Bij.shape[0], Aij.shape[-1], Bij.shape[-1]).to(Aij.dtype)
            bufferAB = bufferAB.to(device)
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_A_1  = dist.P2POp(dist.isend, Aij, peer=rank+1)
        send_B_2  = dist.P2POp(dist.isend, Bij, peer=rank+2)
        recv_AB_2 = dist.P2POp(dist.irecv, bufferAB, peer=rank+2)
        
        reqs0 = dist.batch_isend_irecv([send_A_1, send_B_2])

        Cij = torch.matmul(Aij.mT, Bij)
        for req in reqs0:
            req.wait()
            
        reqs1 = dist.batch_isend_irecv([recv_AB_2])
        print(f'Here rank 0')
        
        for req in reqs1:
            req.wait()
    elif mod == 1:
        bufferA0 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferA2 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferB3 = get_preallocated_buffer((Bij.shape), Bij.dtype, device)

        send_A_2 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        
        recv_A_0 = dist.P2POp(dist.irecv, bufferA0, peer=rank-1)
        recv_A_2 = dist.P2POp(dist.irecv, bufferA2, peer=rank+1)
        recv_B_3 = dist.P2POp(dist.irecv, bufferB3, peer=rank+2)

        reqs0 = dist.batch_isend_irecv([recv_A_0])
        # Do local computation
        AB = torch.matmul(Aij.mT, Bij)
        send_AB_3 = dist.P2POp(dist.isend, AB, peer=rank+2)
        print(f'Here rank 1')
        for req in reqs0:
            req.wait()
        reqs1 = dist.batch_isend_irecv([send_AB_3])
        
        Cij = torch.matmul(bufferA0.mT, Bij)  
        
        reqs2 = dist.batch_isend_irecv([send_A_2, recv_A_2, recv_B_3])
        for req in reqs2:
            req.wait()
        for req in reqs1:
            req.wait()

    elif mod == 2:
        bufferA1 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferA3 = get_preallocated_buffer((Aij.shape), Aij.dtype, device)
        bufferB0 = get_preallocated_buffer((Bij.shape), Bij.dtype, device)
        
        #send_A_0 = dist.P2POp(dist.isend, Aij, peer=rank-2)
        send_A_1 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        #send_B_0 = dist.P2POp(dist.isend, Bij, peer=rank-2)

        recv_A_1 = dist.P2POp(dist.irecv, bufferA1, peer=rank-1)
        recv_A_3 = dist.P2POp(dist.irecv, bufferA3, peer=rank+1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferB0, peer=rank-2)
        
        reqs0 = dist.batch_isend_irecv([recv_A_3])
        AB = torch.matmul(Aij.mT, Bij)
        send_AB_0 = dist.P2POp(dist.isend, AB, peer=rank-2)

        for req in reqs0:
            req.wait()

        reqs1 = dist.batch_isend_irecv([send_AB_0])
        print(f'here rank 2')

        for req in reqs0:
            req.wait()

        Cij = torch.matmul(bufferA3.mT, Bij) 

        reqs2 = dist.batch_isend_irecv([send_A_1, recv_A_1, recv_B_0])
        for req in reqs1:
            req.wait()
        for req in reqs2:
            req.wait()
    else:
        if len(Aij.shape) > 2:
            bufferAB = get_preallocated_buffer((Aij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        else:
            bufferAB = get_preallocated_buffer((Bij.shape[0], Aij.shape[-1], Bij.shape[-1]), Aij.dtype, device)
        
        send_A_2  = dist.P2POp(dist.isend, Aij, peer=rank-1)
        send_B_1  = dist.P2POp(dist.isend, Bij, peer=rank-2)
        recv_AB_1 = dist.P2POp(dist.irecv, bufferAB, peer=rank-2)

        reqs0 = dist.batch_isend_irecv([send_A_2, send_B_1])

        Cij = torch.matmul(Aij.mT, Bij)

        for req in reqs0:
            req.wait()
        print(f'here rank 3')
        
        reqs1 = dist.batch_isend_irecv([recv_AB_1])

        for req in reqs1:
            req.wait()
    
    if mod == 0:
        Cij = Cij + bufferAB #torch.matmul(bufferA2.mT, bufferB2)    
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferA2.mT, bufferB3) 
    elif mod == 2:
        Cij = Cij + torch.matmul(bufferA1.mT, bufferB0) 
    else:
        Cij = Cij + bufferAB
    print(f'Rank {rank} made it')
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
        bufferB = torch.zeros(Bij.shape).to(torch.float32).to(device)
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    dist.send(Aij, dst=target)    
                    A_k = Aij
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        buffer  = torch.zeros(Aij.shape).to(torch.float32).to(device)
                        req = dist.recv(buffer, src=root_rank)
                        #req.wait()
                        A_k = buffer


        B_k = distribute_B(Bij, k, rank, bufferB)

        if Cij is None:
            Cij = A_k.mT @ B_k
        else:
            Cij = Cij + A_k.mT @ B_k

    return Cij

def AB_twoway(Aij, Bij, device, rank):
    #Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    #mod = rank % 2
    #B0, B1 = torch.split(Bij, Bij.shape[-1]//2, dim=-1)
    #bufferB = get_preallocated_buffer(B0.shape, Bij.dtype, device)
    #
    #if mod == 0: 
    #    Bsend = torch.matmul(Aij, B1)    
    #    # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
    #    send_B_1 = dist.P2POp(dist.isend, Bsend, peer=rank+1)
    #    recv_B_1 = dist.P2POp(dist.irecv, bufferB, peer=rank+1)
    #    reqs = dist.batch_isend_irecv([send_B_1, recv_B_1])
    #    
    #elif mod == 1:
    #    Bsend = torch.matmul(Aij, B0)    
    #    send_B_0 = dist.P2POp(dist.isend, Bsend, peer=rank-1)
    #    recv_B_0 = dist.P2POp(dist.irecv, bufferB, peer=rank-1)
    #    reqs = dist.batch_isend_irecv([send_B_0, recv_B_0])
    #
    #Cij = torch.matmul(Aij, Bij)
    #
    #for req in reqs:
    #    req.wait()
#
    #Cij = Cij + bufferB
    #
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 2
    B0, B1 = torch.split(Bij, Bij.shape[-2]//2, dim=-2)
    
    
    bufferA = get_preallocated_buffer(Aij.shape, Aij.dtype, device) 
    
    
    if mod == 0: 
    
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_B_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        reqs = dist.batch_isend_irecv([send_B_1, recv_B_1])
        Cij = torch.matmul(Aij, B0)
    elif mod == 1:
           
        send_B_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_B_0, recv_B_0])
        Cij = torch.matmul(Aij, B1)
    
    for req in reqs:
        req.wait()
    
    if mod == 0:
        Cij = Cij + torch.matmul(bufferA, B1)
    elif mod == 1:
        Cij = Cij + torch.matmul(bufferA, B0)
    return Cij 

def ABT_twoway(Aij, Bij, device, rank):
    Cij = None 
    mod = rank % 2
    
    B0, B1 = torch.split(Bij, Bij.shape[-2]//2, dim=-2)
    #if mod == 0:
    #    B0 = B0.fill_(1.)
    #    B1 = B1.fill_(0.)
    #else:
    #    B0 = B0.fill_(0.)
    #    B1 = B1.fill_(1.)
    bufferAB = get_preallocated_buffer((torch.matmul(Aij,B0.mT)).shape, Aij.dtype, device) # TODO
    
    if mod == 0: 
        ab = torch.matmul(Aij, B1.mT)
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_B_1 = dist.P2POp(dist.isend, ab, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferAB, peer=rank+1)
        reqs = dist.batch_isend_irecv([send_B_1, recv_B_1])
        Cij = torch.matmul(Aij, B0.mT)
    elif mod == 1:
        ab = torch.matmul(Aij, B0.mT)
        send_B_0 = dist.P2POp(dist.isend, ab, peer=rank-1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferAB, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_B_0, recv_B_0])
        Cij = torch.matmul(Aij, B1.mT)
    for req in reqs:
        req.wait()

    if mod == 0:
        Cij = Cij + bufferAB
    else:
        Cij = Cij + bufferAB
    
    return Cij

def ATB_twoway(Aij, Bij, device, rank):
    
    Cij = None #torch.zeros(*Aij.shape[:-1], Bij.shape[-1]).to(device)
    mod = rank % 2
    #A0, A1 = torch.split(Aij, Aij.shape[-2]//2, dim=-2)
    #B0, B1 = torch.split(Bij, Bij.shape[-2]//2, dim=-2)

    bufferA = get_preallocated_buffer(Aij.shape, Aij.dtype, device)
    
    if mod == 0: 
        # Rank 0 send A, B to 1 and 2. Receives A, B from 1 and 2.
        send_B_1 = dist.P2POp(dist.isend, Aij, peer=rank+1)
        recv_B_1 = dist.P2POp(dist.irecv, bufferA, peer=rank+1)
        reqs = dist.batch_isend_irecv([send_B_1, recv_B_1])
        Cij_top = torch.matmul(Aij.mT, Bij)
    elif mod == 1:
           
        send_B_0 = dist.P2POp(dist.isend, Aij, peer=rank-1)
        recv_B_0 = dist.P2POp(dist.irecv, bufferA, peer=rank-1)
        reqs = dist.batch_isend_irecv([send_B_0, recv_B_0])
        Cij_bottom = torch.matmul(Aij.mT, Bij)
    
    for req in reqs:
        req.wait()
    
    if mod == 0:
        Cij_bottom = torch.matmul(bufferA.mT, Bij)
    else:
        Cij_top = torch.matmul(bufferA.mT, Bij)
    

    Cij = torch.concat([Cij_top, Cij_bottom], dim=-2)
    return Cij
