import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import math

def worker(rank, size, N, P):
    """Worker function for each process."""
    # Initialize the process group
    print(f"initiating {rank}, {size}")
    dist.init_process_group(
        backend='gloo',
        rank=rank,
        world_size=size
    )
    
    # Determine the row and column indices in the process grid
    row = rank // P
    col = rank % P

    # Submatrix size
    sub_N = N // P

    # Create the global matrices A and B (same in all processes)
    A = torch.arange(N*N).reshape(N, N).float()
    B = torch.arange(N*N).reshape(N, N).float()

    # Extract local submatrices
    A_local = A[row * sub_N : (row + 1) * sub_N, :]
    B_local = B[:, col * sub_N : (col + 1) * sub_N]

    # Initialize local result matrix C_local
    C_local = torch.zeros(sub_N, sub_N)
    print(f"my rank: {rank}")
    # Create process groups for row and column communications
    row_group_ranks = [(row * P + i) % size for i in range(P)]
    col_group_ranks = [(i * P + col) % size for i in range(P)]
    row_group = dist.new_group(row_group_ranks)
    col_group = dist.new_group(col_group_ranks)
    print("row group", row_group_ranks)
    print("col group", col_group_ranks)
    dist.barrier()
    for k in range(P):
        print(k)
        # Broadcast submatrix of A along the row
        A_k = torch.zeros(sub_N, sub_N)
        A_src_rank = row * P + k
        if col == k:
            A_k = A_local[:, k * sub_N : (k + 1) * sub_N].clone()
        print(f"{rank}: broadcasting", A_k)
        dist.broadcast(A_k, src=A_src_rank, group=row_group)
        print(f"{rank}: broadcasted", A_k)
        # Broadcast submatrix of B along the column
        B_k = torch.zeros(sub_N, sub_N)
        B_src_rank = k * P + col
        if row == k:
            B_k = B_local[k * sub_N : (k + 1) * sub_N, :].clone()
        dist.broadcast(B_k, src=B_src_rank, group=col_group)

        # Perform local multiplication and accumulate
        C_local += torch.mm(A_k, B_k)

    # Output the local result matrix
    print(f"Rank {rank} finished computation. C_local:\n{C_local}")

    # Cleanup
    dist.destroy_process_group()
    return

def main():
    # Total number of processes
    size = 4  # Must be a perfect square
    P = int(math.sqrt(size))
    N = 4  # Size of the global matrices (N x N)
    print("activated once")
    # Spawn processes
    rank = int(os.getenv("SLURM_PROCID"))       # Get individual process ID.
    world_size = int(os.getenv("SLURM_NTASKS")) # Get overall number of processes.
    slurm_localid = int(os.getenv("SLURM_LOCALID"))

    # Initialize GPUs and dataloaders
    device = f"cpu:{slurm_localid}"
    print(f"rank: {rank}")
    worker(rank, world_size, N, P)

if __name__ == "__main__":
    main()
