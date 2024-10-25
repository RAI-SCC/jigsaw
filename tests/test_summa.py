import summa_dist
import torch
import utils
import pytest

# Global init
comm, _, _ = utils.init_mpi()
rank, world_size = comm.Get_rank(), comm.Get_size()
    

def AB(A, B, rank=0, world_size=4, comm=None):
    C = summa_dist.mul_summa(A, B, rank=rank, world_size=world_size, comm=comm)
    return C

def get_matrix(a, b):
    return torch.normal(mean=0, std=1, size=(a, b))


#@pytest.mark.mpi
def test_AB():
    m, n, p = 4, 2, 3
    A, B = get_matrix(m, n), get_matrix(n, p)
  
    Cij = AB(A, B, rank=rank, world_size=world_size, comm=comm)
  
    A = summa_dist.gather(A, comm, world_size)
    B = summa_dist.gather(B, comm, world_size)
    C = summa_dist.gather(Cij, comm, world_size)

    if rank == 0:
        assert (torch.isclose(C, A @ B).all())
    
def test_ABT():
    m, n, p = 4, 2, 3
    A, B = get_matrix(m, n), get_matrix(p, n)
    Cij = summa_dist.ABT(A, B, rank, world_size, comm)
    A = summa_dist.gather(A, comm, world_size)
    B = summa_dist.gather(B, comm, world_size)
    C = summa_dist.gather(Cij, comm, world_size)
  
    if rank == 0:
        assert torch.isclose(C, A@B.T).all()
    
    
def test_ATB():
    m, n, p = 4, 2, 3
   
    A, B = get_matrix(n, m), get_matrix(n, p)
    Cij = summa_dist.ATB(A, B, rank, world_size, comm)
    A = summa_dist.gather(A, comm, world_size)
    B = summa_dist.gather(B, comm, world_size)
    C = summa_dist.gather(Cij, comm, world_size)
  
    if rank == 0:
        assert torch.isclose(C, A.T@B).all()


