import conftest

import torch
import torch.distributed as dist

import MatMul.src.Pytorch.Utility.MatrixManager as MatrixManager
import MatMul.src.Pytorch.Methods.MGR as MGR

def test_MGR_XW(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # mesh = conftest.make_process_mesh(grid)

    # row = mesh.get_group("row")
    # col = mesh.get_group("col")

    # coords = mesh.get_coordinate()

    dist.barrier()
    if rank == 0:
        print("Setup done!\nReading matrices...")

    # matrix shape
    n = 80
    k = 200
    m = 32

    # read data
    X = MatrixManager.ReadLine(r"MatMul/matrix/X.dat", rank, n // world_size, n, k, isRows=True)
    W = MatrixManager.ReadLine(r"MatMul/matrix/W.dat", rank, m // world_size, m, k, isRows=False)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nRunning multiplication...")

    # multiplication
    X_W = MGR.X_W(X, W, device)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4

    ex_X_W = MatrixManager.ReadLine(r"MatMul/matrix/X_W.dat", rank, n // world_size, n, m)
    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)

def test_MGR_XTW(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # mesh = conftest.make_process_mesh(grid)

    # row = mesh.get_group("row")
    # col = mesh.get_group("col")

    # coords = mesh.get_coordinate()

    dist.barrier()
    if rank == 0:
        print("Setup done!\nReading matrices...")

    # matrix shape
    n = 80
    k = 200
    m = 32

    # read data
    X = MatrixManager.ReadLine(r"MatMul/matrix/XT.dat", rank, k // world_size, k, n, isRows=True)
    W = MatrixManager.ReadLine(r"MatMul/matrix/W.dat", rank, m // world_size, m, k, isRows=False)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nRunning multiplication...")

    # multiplication
    X_W = MGR.XT_W(X, W, device)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4

    ex_X_W = MatrixManager.ReadLine(r"MatMul/matrix/XT_W.dat", rank, n // world_size, n, m)
    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)

def test_MGR_XWT(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # mesh = conftest.make_process_mesh(grid)

    # row = mesh.get_group("row")
    # col = mesh.get_group("col")

    # coords = mesh.get_coordinate()

    dist.barrier()
    if rank == 0:
        print("Setup done!\nReading matrices...")

    # matrix shape
    n = 80
    k = 200
    m = 32

    # read data
    X = MatrixManager.ReadLine(r"MatMul/matrix/X.dat", rank, n // world_size, n, k, isRows=True)
    W = MatrixManager.ReadLine(r"MatMul/matrix/WT.dat", rank, k // world_size, k, m, isRows=False)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nRunning multiplication...")

    # multiplication
    X_W = MGR.X_WT(X, W, device)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4

    ex_X_W = MatrixManager.ReadLine(r"MatMul/matrix/X_WT.dat", rank, n // world_size, n, m)
    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)