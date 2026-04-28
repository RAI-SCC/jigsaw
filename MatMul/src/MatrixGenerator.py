import torch
import os

def generate_matrices(paths=r"MatMul/matrix", low=-1, high=1, n=800, k=2000, m=1600, seed=0):
    torch.manual_seed(seed)
    os.makedirs(paths, exist_ok=True)

    X = torch.rand((n, k), dtype=torch.float32) * (high - low) + low
    W = torch.rand((k, m), dtype=torch.float32) * (high - low) + low
    XT = torch.rand((k, n), dtype=torch.float32) * (high - low) + low
    WT = torch.rand((m, k), dtype=torch.float32) * (high - low) + low

    data_map = {
        "X.dat": X, "W.dat": W, "XT.dat": XT, "WT.dat": WT,
        "X_W.dat": X @ W, "XT_W.dat": XT.t() @ W, "X_WT.dat": X @ WT.t()
    }

    for filename, tensor in data_map.items():
        with open(os.path.join(paths, filename), "wb") as f:
            f.write(tensor.numpy().tobytes())

if __name__ == "__main__":
        generate_matrices(n=80, k=200, m=32)