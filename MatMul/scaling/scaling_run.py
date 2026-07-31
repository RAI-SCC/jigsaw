import torch
import torch.distributed as dist

import os, sys

sys.path.append(r"/p/home/jusers/pettenuzzo1/juwels/jigsaw")
print(sys.path)

import MatMul.src.Pytorch.Utility.MatrixManager as MatrixManager
import MatMul.src.Pytorch.Utility.ProcManager as ProcManager

import MatMul.src.Pytorch.Methods.SUMMA as SUMMA
import MatMul.src.Pytorch.Methods.Cannon as Cannon
import MatMul.src.Pytorch.Methods.MGR as MGR

import time


# matrix shape
n = int(sys.argv[1])
k = n
m = n

def scaling():
    ############################# setup
    """Initialize the distributed process group once for the entire test session."""
    rank = int(os.getenv("SLURM_PROCID", -1))
    world_size = int(os.getenv("SLURM_NTASKS", -1))
    local_id = int(os.getenv("SLURM_LOCALID", rank))

    device_id = local_id if torch.cuda.device_count() != 1 else 0
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
        init_method="env://",
    )

    ############################# setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid = (grid_size, grid_size)

    coords = ProcManager.GetCoords(rank, grid)

    if rank == 0:
        results = {}

    dist.barrier()

    ############################# Generate matrices
    X = torch.rand(n, k) * 2 - 1
    W = torch.rand(k, m) * 2 - 1
    XT = torch.rand(k, n) * 2 - 1
    WT = torch.rand(m, k) * 2 - 1

    X = X.to(device=device)
    W = W.to(device=device)
    XT = XT.to(device=device)
    WT = WT.to(device=device)

    ############################# SUMMA
    if rank == 0:
        print("Setup done!\nRunning tests for SUMMA...")

    dist.barrier()
    if rank == 0:
        print("Creating communicators...")

    comm = SUMMA.ComputeGroups(grid)

    if rank == 0:
        print("Communicators created!")

    # multiplications
    if rank == 0:
        print("Running SUMMA X@W...")
        s_t = time.perf_counter()
    
    SUMMA.X_W(X, W, grid, device, comm)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["SUMMA_X@W"] = e_t - s_t

        print("Done!")

    if rank == 0:
        print("Running SUMMA X^T@W...")
        s_t = time.perf_counter()
    
    SUMMA.XT_W(XT, W, grid, device, comm)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["SUMMA_X^T@W"] = e_t - s_t

        print("Done!")

    if rank == 0:
        print("Running SUMMA X@W^T...")
        s_t = time.perf_counter()
    
    SUMMA.X_WT(X, WT, grid, device, comm)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["SUMMA_X@W^T"] = e_t - s_t

        print("Done!")
    
    ############################# Cannon
    if rank == 0:
        print("Running tests for Cannon...")

    dist.barrier()

    # multiplications
    if rank == 0:
        print("Running Cannon X@W...")
        s_t = time.perf_counter()
    
    Cannon.X_W(X, W, grid, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["Cannon_X@W"] = e_t - s_t

        print("Done!")

    if rank == 0:
        print("Running Cannon X^T@W...")
        s_t = time.perf_counter()
    
    Cannon.XT_W(XT, W, grid, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["Cannon_X^T@W"] = e_t - s_t

        print("Done!")
    
    if rank == 0:
        print("Running Cannon X@W^T...")
        s_t = time.perf_counter()
    
    Cannon.X_WT(X, WT, grid, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["Cannon_X@W^T"] = e_t - s_t

        print("Done!")

    # ############################# MGR
    if rank == 0:
        print("Running tests for MGR...")

    dist.barrier()

    # multiplications
    if rank == 0:
        print("Running MGR X@W...")
        s_t = time.perf_counter()
    
    MGR.X_W(X, W, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["MGR_X@W"] = e_t - s_t

        print("Done!")

    if rank == 0:
        print("Running MGR X^T@W...")
        s_t = time.perf_counter()
    
    MGR.XT_W(XT, W, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["MGR_X^T@W"] = e_t - s_t

        print("Done!")
    
    if rank == 0:
        print("Running MGR X@W^T...")
        s_t = time.perf_counter()
    
    MGR.X_WT(X, WT, device)
    
    dist.barrier()

    if rank == 0:
        e_t = time.perf_counter()

        results["MGR_X@W^T"] = e_t - s_t

        print("Done!")

    if rank == 0:
        print(f"{world_size}-{n}: {results}")

if __name__ == "__main__":
    print("Starting...")

    try:
        scaling()
    finally:
        dist.destroy_process_group()