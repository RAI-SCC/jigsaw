import os
import sys

import torch.multiprocessing as mp
import torch.distributed as dist

import Pytorch.Utility.MatrixManager as MatrixManager
import Pytorch.Utility.ProcManager as ProcManager

def run(rank, world_size):
    # 1. Standard distributed setup
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    # os.environ['GLOO_SOCKET_IFNAME'] = socket.gethostname()
    
    # 2. Initialize the process group
    # We pass rank and world_size directly to avoid env var lookups
    print(f"Subprocess {rank} initializing...")
    dist.init_process_group(
        backend="gloo", 
        init_method="tcp://127.0.0.1:12355", 
        rank=rank, 
        world_size=world_size
    )
    
    print(f"Rank {rank} initialized.")

    dist.barrier()

    # Block utilities check
    coords = ProcManager.GetCoords(rank, (2, 2))

    W = MatrixManager.Read(r"matrix/W.dat", coords, 75, 25, 2, 2)

    MatrixManager.Write(W, coords, 2, 2, r"matrix/results.dat")

    X = MatrixManager.Read(r"matrix/results.dat", coords, 75, 25, 2, 2)

    res = MatrixManager.Compare(W, X)

    if rank == 0:
        print("Comparation: ", res)

    # MGR utilities check
    W = MatrixManager.ReadLine(r"matrix/W.dat", rank, 150 // 4, 150, 50)

    MatrixManager.WriteLine(W, r"matrix/results.dat", rank, 150 // 4, 150, 50)

    X = MatrixManager.ReadLine(r"matrix/results.dat", rank, 150 // 4, 150, 50)

    res = MatrixManager.Compare(W, X)

    if rank == 0:
        print("Comparation: ", res)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4

    try:
        world_size = int(sys.argv[1])
    except Exception as e:
        print(f"{e}\n\nDefaulting to {world_size} processors")

    print(f"Spawning {world_size} processes...")
    
    try:
        mp.spawn(
            run,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Spawn failed: {e}")