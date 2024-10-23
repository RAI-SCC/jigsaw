from mpi4py import MPI
import numpy as np
import torch

def canons_algo():
    n = 8
    A = torch.arange(n*n).reshape(n, n)
    B = torch.arange(n*n).reshape(n, n)
    p = 4 # = world_size
    sqrtp = int(np.sqrt(p))
    subset_size = int(n//np.sqrt(p))
    C = torch.zeros((n, n))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() # rank is from 0 to 3

    i = int(rank // np.sqrt(p)) # 0/2
    j = int(rank % np.sqrt(p))

    i_start = (i - sqrtp + j) % sqrtp
    j_start = (j - sqrtp + i) % sqrtp

    Aij = torch.tensor(A[i*subset_size:(i+1)*subset_size, j_start*subset_size:(j_start+1)*subset_size].clone().detach()).to(torch.float32)
    Aij.requires_grad = True
    Bij = torch.tensor(B[(i_start)*subset_size:(i_start+1)*subset_size, j*subset_size:(j+1)*subset_size].clone().detach()).to(torch.float32)

    Cij = torch.matmul(Aij, Bij)
    #Cij = np.zeros((subset_size, subset_size))
    i_target = (i - 1) % sqrtp
    j_target = (j - 1) % sqrtp

    A_target = int((i*sqrtp + j_target%sqrtp)%p)
    B_target = int((i_target*sqrtp + j)%p)

    A_src = int((i*sqrtp + (j+1) % sqrtp)%p)
    B_src = int(((i+1)*sqrtp + j) % p)

    for step in range(sqrtp-1):
        # Send 
        comm.isend(Aij, dest=A_target, tag=13)
        comm.isend(Bij, dest=B_target, tag=14)

        Aij_data = torch.zeros(subset_size, subset_size)

        req = comm.irecv(source=A_src, tag=13)
        req2 = comm.irecv(source=B_src, tag=14)
        Aij_data = req.wait()
        Bij_data = req2.wait()

        Cij = Cij + torch.matmul(Aij_data, Bij_data)


    C[i*subset_size:(i+1)*subset_size, j*subset_size:(j+1)*subset_size] = Cij

    if rank == 0:
        print(torch.matmul(A, B))

    print(C)


def init_summa_algo(n, i, j, world_size):
    # n: side-length of matrix
    # i, j: row and column of rank/block matrix
    # world_size
    A = torch.arange(n*n).reshape(n, n).to(torch.float32)
    B = torch.arange(n*n).reshape(n, n).to(torch.float32) 
    p = world_size
    
    subset_size = int(n//np.sqrt(p))
    

    
    Aij = A[i*subset_size:(i+1)*subset_size, j*subset_size:(j+1)*subset_size].clone().detach()
    Bij = B[i*subset_size:(i+1)*subset_size, j*subset_size:(j+1)*subset_size].clone().detach()
    
    Cij = torch.zeros(subset_size, subset_size)

    return Aij, Bij, Cij, A, B

def init_mpi():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # rank is from 0 to 3
    
    p = comm.Get_size()
    i = int(rank // np.sqrt(p)) # 0/2
    j = int(rank % np.sqrt(p))
    return comm, i, j


def mul_summa(Aij, Bij, Cij, n, rank, world_size, comm):

    C = torch.zeros((n, n))
    
    sqrtp = int(np.sqrt(world_size))
    subset_size = int(n//np.sqrt(world_size))
    
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    
    # Create process groups for row and column communications
    row_group_ranks = [i * sqrtp + k for k in range(sqrtp)]
    col_group_ranks = [k * sqrtp + j for k in range(sqrtp)]

    #row_group = dist.new_group(row_group_ranks)
    #col_group = dist.new_group(col_group_ranks)
    row_group = comm.group.Incl(row_group_ranks)
    row_comm  = comm.Create_group(row_group)
    
    col_group  = comm.group.Incl(col_group_ranks)
    col_comm = comm.Create_group(col_group)
    # Empty allocation of A_k per process
    A_k = torch.zeros(subset_size, subset_size)

    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        
        if j == k: # Broadcast horizontally
            A_k = Aij.clone().detach()  # i = 0; --> 0, 3, 6
        #root = (j*sqrtp+k)%p
        A_k = row_comm.bcast(A_k, root=k) 

        # Broadcast submatrix of B along the column
        B_k = torch.zeros(subset_size, subset_size)
        
        if i == k: # broadcast across rows
            B_k = Bij.clone().detach()
        
        B_k = col_comm.bcast(B_k, root=k)
        
        ## Perform local multiplication and accumulate
        Cij = Cij + torch.matmul(A_k, B_k)
        
    return Cij
    
def transpose(n, rank):
    C = torch.arange(n*n).reshape(n, n).T.reshape(-1)
    transpose_index = int(C[rank])
    return transpose_index
    


def ABT_manual(Aij, Bij, Cij, n, rank, world_size, comm):
    C = torch.zeros((n, n))
    
    sqrtp = int(np.sqrt(world_size))
    subset_size = int(n//np.sqrt(world_size))
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
   
    
    # Create process groups for row and column communications
    row_group_ranks = [i * sqrtp + k for k in range(sqrtp)]
    row_group = comm.group.Incl(row_group_ranks)
    row_comm  = comm.Create_group(row_group)

    print(f"rank {rank}:, {row_group_ranks}")
    
 
    # Empty allocation of A_k per process
    A_k = torch.zeros(subset_size, subset_size)
    #
    #print(f"Real rank {rank}, B value {Bij}, j value {jb}, i value {ib}")

    
    # Broadcast submatrix of A along the row
    if j == 0: # Broadcast horizontally
        A_k = Aij.clone().detach()  # i = 0; --> 0, 3, 6
    #root = (j*sqrtp+k)%p
    #A_k = Aij.clone().detach()
    B_k = Bij.clone().detach()

    A_k = row_comm.bcast(A_k, root=0) 


# SYNTAX    comm.isend(Bij, dest=B_target, tag=14)
     #for k = 0:
    if rank == 0:
        comm.isend(Bij, dest=2, tag=14)
    if rank == 1:
        req2 = comm.irecv(source=2, tag=16)
        B_k = req2.wait()
        
    if rank == 2:
        req = comm.irecv(source=0, tag=14)
        B_k = req.wait()
        comm.isend(Bij, dest=1, tag=16)
        comm.isend(Bij, dest=3, tag=17)
    if rank == 3:
        req2 = comm.irecv(source=2, tag=17)
        B_k = req2.wait()
    
    Cij += A_k @ B_k.T

    A_k = torch.zeros(subset_size, subset_size)

    if j == 1:
        A_k = Aij.clone().detach()  # i = 0; --> 0, 3, 6
    A_k = row_comm.bcast(A_k, root=1) 

    # k = 1
    if rank == 0:
        req2 = comm.irecv(source=1, tag=13)
        B_k = req2.wait()
    if rank == 1:
        comm.isend(Bij, dest=0, tag=13)        
        comm.isend(Bij, dest=2, tag=14)     
        req = comm.irecv(source=3, tag=10)
        B_k = req.wait()
    if rank == 2:
        req2 = comm.irecv(source=1, tag=14)
        B_k = req2.wait()
    if rank == 3:
        comm.isend(Bij, dest=1, tag=10)
        B_k = Bij
        
    Cij += A_k @ B_k.T
    return Cij

def verify(A, B, Cij, comm, world_size, n):
    C_mul = A @ B
    
    rank = comm.Get_rank()
    p = int(np.sqrt(world_size))

    C_collective = comm.gather(Cij, root=0)
    if rank == 0:
        C_collective = torch.stack(C_collective)
        C_collective = C_collective.reshape(p, p, *C_collective.shape[1:])
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(n, n)
        
        print(C_collective == C_mul)
    


def verifyT(A, B, Cij, comm, world_size, n):
    C_mul = A @ B.T
    C     = torch.zeros_like(C_mul)
    
    rank = comm.Get_rank()
    p = int(np.sqrt(world_size))
    
    C_collective = comm.gather(Cij, root=0)
    if rank == 0:
        C_collective = torch.stack(C_collective)
        C_collective = C_collective.reshape(p, p, *C_collective.shape[1:])
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(n, n)
        
        print("Collective", C_collective)
        print("real", A @ B.T)

def ABT(Aij, Bij, Cij, n, rank, world_size, comm):
    sqrtp = int(np.sqrt(world_size))
    subset_size = int(n//np.sqrt(world_size))
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    
    # Create process groups for row and column communications
    row_group_ranks = [i * sqrtp + k for k in range(sqrtp)]
    row_group = comm.group.Incl(row_group_ranks)
    row_comm  = comm.Create_group(row_group)

    rank_dict_transpose = {}
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
    
    if rank == 0:
        print(rank_dict_transpose)

    A_k = torch.zeros(subset_size, subset_size)
    
    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        if j == k: # Broadcast horizontally
            A_k = Aij.clone().detach()  # i = 0; --> 0, 3, 6

        B_k = Bij.clone().detach()

        A_k = row_comm.bcast(A_k, root=k) 

        # for k = 0
        # "root" ranks:
        root_ranks = np.arange(sqrtp)*sqrtp + k

        # sending
        if rank in root_ranks:
            targets = rank_dict_transpose[rank]
            for target in targets: 
                if rank != target:
                    comm.isend(Bij, dest=target, tag=14)    
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        req = comm.irecv(source=root_rank, tag=14)
                        B_k = req.wait()
    
        Cij += A_k @ B_k.T

    return Cij

n = 4           # problem size
comm, i, j = init_mpi()
world_size = comm.Get_size()
Aij, Bij, Cij, A, B = init_summa_algo(n, i, j, world_size)

Cij = ABT(Aij, Bij, Cij, n, comm.Get_rank(), world_size, comm)

verifyT(A, B, Cij, comm, world_size, n)