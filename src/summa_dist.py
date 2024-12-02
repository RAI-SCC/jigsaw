from mpi4py import MPI
import numpy as np
import torch

def init_summa_algo(m, n, p, i, j, world_size):
    # n: side-length of matrix
    # i, j: row and column of rank/block matrix
    # world_size
    #A = torch.arange(m*n).reshape(m, n).to(torch.float32)
    #B = torch.arange(n*p).reshape(n, p).to(torch.float32)
    torch.manual_seed(19)
    A = torch.tensor([1, 1, 3, 0, 1, 0, 0, 1]).reshape(m, n).to(torch.float32)
    #A = torch.rand(m*n).reshape(m,n).to(torch.float32)   
    #B = torch.rand(n*p).reshape(n, p).to(torch.float32)   
    B = torch.normal(0, 1, size=(n, p))
    
    subset_size_m = int(m//np.sqrt(world_size))
    subset_size_n = int(n//np.sqrt(world_size))
    subset_size_p = int(p//np.sqrt(world_size))
    
    Aij = A[i*subset_size_m:(i+1)*subset_size_m, j*subset_size_n:(j+1)*subset_size_n].clone().detach()
    Bij = B[i*subset_size_n:(i+1)*subset_size_n, j*subset_size_p:(j+1)*subset_size_p].clone().detach()
    
    Cij = torch.zeros(subset_size_m, subset_size_p)

    return Aij, Bij, Cij, A, B


def init_summa_vert(m, n, rank):
    # n: side-length of matrix
    # i, j: row and column of rank/block matrix
    # world_size
    A = torch.tensor([1, 2, 3, 4, 0, 3, 4, 5]).reshape(m, n).to(torch.float32)
    Bij = torch.normal(0, 1, size=(1, 2))

    Aij = A[:, rank]
    
    if rank == 0:
        Cij = torch.tensor([20, 15, 23, 16, 55, 50, 15, 10]).reshape(4, 2)
    Cij = torch.zeros(1, 1)

    return Aij, Bij, Cij, A


def mul_summa(Aij, Bij, rank, world_size, comm):

    assert(Aij.shape[1] == Bij.shape[0])
    m, n, p = Aij.shape[0], Aij.shape[1], Bij.shape[1] 
       
    Cij = torch.zeros(m, p)
    
    sqrtp = int(np.sqrt(world_size))
    subset_size_m, subset_size_n, subset_size_p = m//sqrtp, n//sqrtp, p//sqrtp
    
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    
    # Create process groups for row and column communications
    row_group_ranks = [i * sqrtp + k for k in range(sqrtp)]
    col_group_ranks = [k * sqrtp + j for k in range(sqrtp)]

    row_group = comm.group.Incl(row_group_ranks)
    row_comm  = comm.Create_group(row_group)
    
    col_group  = comm.group.Incl(col_group_ranks)
    col_comm = comm.Create_group(col_group)
    # Empty allocation of A_k per process
    A_k = torch.zeros(subset_size_m, subset_size_n)

    for k in range(sqrtp):
        # Broadcast submatrix of A along the row
        
        if j == k: # Broadcast horizontally
            A_k = Aij.clone()  # i = 0; --> 0, 3, 6
        #root = (j*sqrtp+k)%p
        A_k = row_comm.bcast(A_k, root=k) 

        # Broadcast submatrix of B along the column
        B_k = torch.zeros(subset_size_n, subset_size_p)
        
        if i == k: # broadcast across rows
            B_k = Bij.clone()
        
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
    B_k = Bij.clone().detach()

    A_k = row_comm.bcast(A_k, root=0) 


    
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
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(C_mul.shape[0], C_mul.shape[1])
        
        #print("Collective", C_collective)
        #print("real", A @ B.T)

        return C_collective
        
def verifyATB(A, B, Cij, comm, world_size, n):
    C_mul = A.T @ B
    C     = torch.zeros_like(C_mul)
    
    rank = comm.Get_rank()
    p = int(np.sqrt(world_size))
    
    C_collective = comm.gather(Cij, root=0)

    if rank == 0:
        C_collective = torch.stack(C_collective)
        C_collective = C_collective.reshape(p, p, *C_collective.shape[1:])
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(C_mul.shape[0], C_mul.shape[1])
        
        print("Collective", C_collective)
        print("real", C_mul)

        return C_collective
    
def gather(Cij, comm, world_size):
    rank = comm.Get_rank()
    p = int(np.sqrt(world_size))
    
    C_collective = comm.gather(Cij, root=0)

    if rank == 0:
        C_collective = torch.stack(C_collective)
        C_collective = C_collective.reshape(p, p, *C_collective.shape[1:])
        C_collective = C_collective.permute(0, 2, 1, 3).reshape(Cij.shape[0]*p, Cij.shape[1]*p)

    return C_collective
        

def ABT(Aij, Bij, rank, world_size, comm):
    sqrtp = int(np.sqrt(world_size))
    
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

    A_k = torch.zeros_like(Aij).detach()
    Cij = torch.zeros(Aij.shape[0], Bij.shape[0]).to(A_k.device)
    
    buffer = np.empty_like(Bij.cpu().numpy())

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
                    comm.Isend(Bij.cpu().numpy(), dest=target, tag=30)    
                    
        
        # receiving
        for root_rank in root_ranks:
            if root_rank != rank:
                for target in rank_dict_transpose[root_rank]:
                    if rank == target:
                        req = comm.Irecv(buffer, source=root_rank, tag=30)
                        req.wait()
                        B_k = torch.from_numpy(buffer).to(A_k.device) 


        #print(f"Cij device {Cij.device}, A_k device {A_k.device}, B_k device {B_k.device}")
        Cij = Cij + A_k @ B_k.T


    return Cij

def ATB(Aij, Bij, rank, world_size, comm):
    
    sqrtp = int(np.sqrt(world_size))
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))

    
    # Create process groups for row and column communications
    col_group_ranks = [k * sqrtp + j for k in range(sqrtp)]
    col_group = comm.group.Incl(col_group_ranks)
    col_comm  = comm.Create_group(col_group)

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
    Cij = torch.zeros(Aij.shape[1], Bij.shape[1]).to(B_k.device)


    buffer = np.empty_like(Aij.cpu().numpy())

    
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
                comm.Isend(Aij, dest=target, tag=14)    
        
        # receiving
        for root_rank in root_ranks:
            
            for target in rank_dict_transpose[root_rank]:
                if rank == target:
                    req = comm.Irecv(buffer, source=root_rank, tag=14)
                    req.wait()
                    A_k = torch.from_numpy(buffer).to(B_k.device)

        if i == k: # Broadcast horizontally
            B_k = Bij.clone()  # i = 0; --> 0, 3, 6

        B_k = col_comm.bcast(B_k, root=k) 
    
        Cij = Cij + A_k.T @ B_k
        #if k == 0:
        #    print(f"Rank {rank}, k {k}, A_k {A_k}, B_k {B_k}")

    return Cij

#n = 4           # problem size
#comm, i, j = init_mpi()
#world_size = comm.Get_size()
#Aij, Bij, Cij, A, B = init_summa_algo(m=2, n=2, p=2, i=i, j=j, world_size=world_size)
#Cij = ATB(Aij, Bij, n, comm.Get_rank(), world_size, comm)
#verifyATB(A, B, Cij, comm, world_size, n)