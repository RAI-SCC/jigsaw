import os
import sys

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import Pytorch.Methods.SUMMA as SUMMA
import Pytorch.Methods.Cannon as Cannon
import Pytorch.Methods.MGR as MGR

import Pytorch.Utility.MatrixManager as MatrixManager
import Pytorch.Utility.ProcManager as ProcManager

from torch.profiler import profile, record_function, ProfilerActivity


# Constants
world_size = 16
method = "SUMMA"

tol = 1e-4

x_path = r"matrix/XT.dat"
w_path = r"matrix/W.dat"
x_w_path = r"matrix/XT_W.dat"

n = 800
k = 2000
m = 1600

num_r = 4
num_c = 4
grid_shape = (num_r, num_c)

device = torch.device("cpu")


def run(rank, world_size, method, device):
    # 1. Standard distributed setup
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    
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

    if rank == 0:
        print(f"Running matrix multiplication.")

    # Read matrices
    if method != "MGR":
        coords = ProcManager.GetCoords(rank, (num_r, num_c))

        sub_n, sub_k, sub_m = n // num_r, k // num_r, m // num_c

        # Read matrices
        X = MatrixManager.ReadT(x_path, coords, sub_k, sub_n, num_r, num_c)
        W = MatrixManager.Read(w_path, coords, sub_k, sub_m, num_r, num_c)
    else:
        sub_n = n // world_size
        sub_m = m // world_size
        
        # 1. CARICAMENTO INIZIALE
        # X è (n x k). Carichiamo RIGHE: lines_per_block=sub_n, total_lines=n, other_dim=k
        X = MatrixManager.ReadLine(x_path, rank, sub_n, n, k, isRows=False)

        # W è (k x m). Carichiamo COLONNE: lines_per_block=sub_m, total_lines=m, other_dim=k
        # Nota: W_current cambierà ad ogni step del ring
        W = MatrixManager.ReadLine(w_path, rank, sub_m, m, k, isRows=False)

    X = X.to(device)
    W = W.to(device)

    if method == "SUMMA":
        group = SUMMA.ComputeGroups(grid_shape)

    # Multiplication (+ profiling)
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA if device.type == 'cuda' else ProfilerActivity.CPU
        ],
        record_shapes=True,
        with_stack=True,
        acc_events=True
    ) as prof:
        with record_function("dist_matmul"):
            # MATRIX MULTIPLICATION SECTION
            if method == "SUMMA":
                x_w = SUMMA.XT_W(X, W, grid_shape, device, group)
            elif method == "Cannon":
                x_w = Cannon.XT_W(X, W, grid_shape, device)
            elif method == "MGR":
                x_w = MGR.XT_W(X, W, device)
    

    if rank == 0:
        print("Multiplication done!")
        # Sort by total time and print
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        # Optionally export for Chrome tracing (view at chrome://tracing)
        prof.export_chrome_trace("profiler/trace.json")

    # Final comparison
    if method != "MGR":
        ex = MatrixManager.Read(x_w_path, ProcManager.GetCoords(rank, grid_shape), n // grid_shape[0], m // grid_shape[1], num_r, num_c)
    else:
        ex = MatrixManager.ReadLine(x_w_path, rank, sub_n, n, m)
    
    res = MatrixManager.Compare(x_w, ex, tol)

    if rank == 0:
        print("Final comparison: ", res)

    dist.destroy_process_group()

if __name__ == "__main__":
    try:
        method = sys.argv[1]

        if len(sys.argv) > 2:
            world_size = int(sys.argv[2])
    except Exception as e:
        print(f"{e}\n\nDefaulting to {world_size} processors")

    print(f"Method: {method}\nProcessors: {world_size}\nDevice: {device}")

    try:
        mp.spawn(
            run,
            args=(world_size, method, device),
            nprocs=world_size,
            join=True
        )
    except Exception as e:
        print(f"Spawn failed: {e}")