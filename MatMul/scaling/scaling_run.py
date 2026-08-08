import torch
import torch.distributed as dist

import os, sys

sys.path.append(os.path.expanduser('~/jigsaw'))

from MatMul.scaling.SUMMA import T_SUMMA
from MatMul.scaling.Cannon import T_Cannon
# from MatMul.scaling.MGR import T_MGR

def run_scaling_tests():
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

    ############################# setup rank and grid
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid = (grid_size, grid_size)

    n_iter = 5
    n_warm = 2

    # Define the list of matrix shapes/sizes you want to iterate over
    # matrix_sizes = [256, 512, 1024, 2048, 4096] 
    matrix_sizes = [8192, 16384, 32768]

    ############################# Initialize Benchmark Classes once
    summa_tester = T_SUMMA(grid, device, n_warm, n_iter, rank)
    cannon_tester = T_Cannon(grid, device, n_warm, n_iter, rank)
    # mgr_tester = T_MGR(grid, device, n_warm, n_iter, rank)

    # Iterate over each matrix size
    for n in matrix_sizes:
        k = n
        m = n

        if rank == 0:
            results = {}
            print(f"\n==========================================")
            print(f"Running benchmarks for Matrix Size N = {n}")
            print(f"==========================================")

        dist.barrier()

        # Generate and move matrices to device for the current size
        X = torch.rand(n, k, device=device) * 2 - 1
        W = torch.rand(k, m, device=device) * 2 - 1
        XT = torch.rand(k, n, device=device) * 2 - 1
        WT = torch.rand(m, k, device=device) * 2 - 1

        ############################# SUMMA Tests
        if rank == 0:
            print("--- Running tests for SUMMA ---")

        dist.barrier()
        
        # X@W
        if rank == 0: print("Warming up SUMMA X@W...")
        summa_tester.X_W_warmup(X, W)
        if rank == 0: print("Running timed SUMMA X@W...")
        t = summa_tester.X_W(X, W)
        if rank == 0: results["SUMMA_X@W"] = t

        dist.barrier()

        # XT@W
        if rank == 0: print("Warming up SUMMA X^T@W...")
        summa_tester.XT_W_warmup(XT, W)
        if rank == 0: print("Running timed SUMMA X^T@W...")
        t = summa_tester.XT_W(XT, W)
        if rank == 0: results["SUMMA_X^T@W"] = t

        dist.barrier()

        # X@WT
        if rank == 0: print("Warming up SUMMA X@W^T...")
        summa_tester.X_WT_warmup(X, WT)
        if rank == 0: print("Running timed SUMMA X@W^T...")
        t = summa_tester.X_WT(X, WT)
        if rank == 0: results["SUMMA_X@W^T"] = t

        ############################# Cannon Tests
        if rank == 0:
            print("--- Running tests for Cannon ---")

        dist.barrier()

        # X@W
        if rank == 0: print("Warming up Cannon X@W...")
        cannon_tester.X_W_warmup(X, W)
        if rank == 0: print("Running timed Cannon X@W...")
        t = cannon_tester.X_W(X, W)
        if rank == 0: results["Cannon_X@W"] = t

        dist.barrier()

        # XT@W
        if rank == 0: print("Warming up Cannon X^T@W...")
        cannon_tester.XT_W_warmup(XT, W)
        if rank == 0: print("Running timed Cannon X^T@W...")
        t = cannon_tester.XT_W(XT, W)
        if rank == 0: results["Cannon_X^T@W"] = t

        dist.barrier()

        # X@WT
        if rank == 0: print("Warming up Cannon X@W^T...")
        cannon_tester.X_WT_warmup(X, WT)
        if rank == 0: print("Running timed Cannon X@W^T...")
        t = cannon_tester.X_WT(X, WT)
        if rank == 0: results["Cannon_X@W^T"] = t

        ############################# MGR Tests
        # if rank == 0:
        #     print("--- Running tests for MGR ---")

        # dist.barrier()

        # # X@W
        # if rank == 0: print("Warming up MGR X@W...")
        # mgr_tester.X_W_warmup(X, W)
        # if rank == 0: print("Running timed MGR X@W...")
        # t = mgr_tester.X_W(X, W)
        # if rank == 0: results["MGR_X@W"] = t

        # dist.barrier()

        # # XT@W
        # if rank == 0: print("Warming up MGR X^T@W...")
        # mgr_tester.XT_W_warmup(XT, W)
        # if rank == 0: print("Running timed MGR X^T@W...")
        # t = mgr_tester.XT_W(XT, W)
        # if rank == 0: results["MGR_X^T@W"] = t

        # dist.barrier()

        # # X@WT
        # if rank == 0: print("Warming up MGR X@W^T...")
        # mgr_tester.X_WT_warmup(X, WT)
        # if rank == 0: print("Running timed MGR X@W^T...")
        # t = mgr_tester.X_WT(X, WT)
        # if rank == 0: results["MGR_X@W^T"] = t

        # Print final aggregated results for this matrix size
        if rank == 0:
            print(f"\n[SUMMARY] World Size: {world_size} | Matrix Size N: {n}")
            print(results)

        # Free up memory before next matrix size iteration
        del X, W, XT, WT
        torch.cuda.empty_cache()
        dist.barrier()

        if rank == 0:
            print("Done!")

if __name__ == "__main__":
    print("Starting scaling test suite...")

    try:
        run_scaling_tests()
    finally:
        dist.destroy_process_group()