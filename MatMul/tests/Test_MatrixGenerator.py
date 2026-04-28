import torch
import os
import numpy as np
from MatMul.src.MatrixGenerator import generate_matrices

def test_generate_matrices(tmp_path):
    path, seed, n, k, m = str(tmp_path), 42, 10, 5, 8
    generate_matrices(paths=path, n=n, k=k, m=m, seed=seed)

    # Re-trace the seed to get the expected X @ W result
    torch.manual_seed(seed)
    X = torch.rand((n, k)) * 2 - 1  # Logic for default low=-1, high=1
    W = torch.rand((k, m)) * 2 - 1
    expected_X_W = (X @ W).numpy()

    # Load and verify the actual file produced
    file_path = os.path.join(path, "X_W.dat")
    actual_data = np.fromfile(file_path, dtype=np.float32).reshape(n, m)

    assert os.path.exists(file_path)
    np.testing.assert_allclose(actual_data, expected_X_W, atol=1e-6)