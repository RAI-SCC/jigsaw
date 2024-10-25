from mpi4py import MPI
import numpy as np

def init_mpi():
    comm = MPI.COMM_WORLD

    rank = comm.Get_rank() # rank is from 0 to 3
    world_size = comm.Get_size()
    i = int(rank // np.sqrt(world_size)) # 0/2
    j = int(rank % np.sqrt(world_size))
    return comm, i, j