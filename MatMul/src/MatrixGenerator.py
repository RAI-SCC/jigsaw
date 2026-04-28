import torch
import sys
import struct
import os

if __name__ == "__main__":
    torch.manual_seed(0)

    low = -1
    high = 1
    n, k, m = 800,2000,1600

    # Argument handling
    if len(sys.argv) == 4:
        n, k, m = map(int, sys.argv[1:4])
    elif len(sys.argv) == 6:
        low, high = map(int, sys.argv[1:3])
        n, k, m = map(int, sys.argv[3:6])

    # Ensure directory exists
    os.makedirs("matrix", exist_ok=True)

    print(f"Generating matrices (float32)... Range: [{low}, {high}]")

    # 1. Generate Tensors
    X = torch.rand((n, k), dtype=torch.float32) * (high - low) + low
    W = torch.rand((k, m), dtype=torch.float32) * (high - low) + low

    XT = torch.rand((k, n), dtype=torch.float32) * (high - low) + low
    WT = torch.rand((m, k), dtype=torch.float32) * (high - low) + low

    X_W = X @ W
    XT_W = XT.t() @ W
    X_WT = X @ WT.t()

    # 2. Write to files using the underlying byte buffer
    # .numpy() creates a view, .tobytes() exports the raw binary data
    with open(r"matrix/X.dat", "wb") as f:
        f.write(X.numpy().tobytes())

    with open(r"matrix/W.dat", "wb") as f:
        f.write(W.numpy().tobytes())

    with open(r"matrix/XT.dat", "wb") as f:
        f.write(XT.numpy().tobytes())
    
    with open(r"matrix/WT.dat", "wb") as f:
        f.write(WT.numpy().tobytes())

    with open(r"matrix/X_W.dat", "wb") as f:
        f.write(X_W.numpy().tobytes())
    
    with open(r"matrix/XT_W.dat", "wb") as f:
        f.write(XT_W.numpy().tobytes())

    with open(r"matrix/X_WT.dat", "wb") as f:
        f.write(X_WT.numpy().tobytes())

    print("Matrices generated, checking format...")

    # 3. Verification using your struct logic
    with open(r"matrix/X.dat", "rb") as f:
        data = f.read()
    
    # struct.iter_unpack returns a generator of tuples; we extract the first element
    unpacked_data = [val[0] for val in struct.iter_unpack("f", data)]
    rX = torch.tensor(unpacked_data).reshape(n, k)

    # Check if they match
    res = torch.all(torch.abs(X - rX) < 1e-6)

    print(f"X - rX -> 0: {res}")
    print(f"X shape: {X.shape}")
    print(f"W shape: {W.shape}")
    print(f"XT shape: {XT.shape}")
    print(f"WT shape: {WT.shape}")
    print(f"X_W shape: {X_W.shape}")
    print(f"XT_W shape: {XT_W.shape}")
    print(f"X_WT shape: {X_WT.shape}")