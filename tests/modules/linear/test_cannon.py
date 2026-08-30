import tests.conftest

import torch
import torch.distributed as dist

import tests.modules.linear.MatrixManager as MatrixManager
import jigsaw.modules.linear.ProcManager as ProcManager

import jigsaw.modules.linear.cannon as Cannon

matrix_path = r"/p/scratch/hai_1302/matrix_test/"

def test_Cannon_XW(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid = (grid_size, grid_size)

    coords = ProcManager.GetCoords(rank, grid)

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
    X = MatrixManager.Read(matrix_path + "X.dat", coords, n // grid_size, k // grid_size, grid_size, grid_size)
    W = MatrixManager.Read(matrix_path + "W.dat", coords, k // grid_size, m // grid_size, grid_size, grid_size)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nRunning multiplication...")

    # multiplication
    X_W = Cannon.x_w_cannon(X, W, dist.get_rank(), device, None)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4
    ex_X_W = MatrixManager.Read(matrix_path + "X_W.dat", coords, n // grid_size, m // grid_size, grid_size, grid_size)

    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)

def test_Cannon_XTW(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid = (grid_size, grid_size)

    coords = ProcManager.GetCoords(rank, grid)

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
    X = MatrixManager.Read(matrix_path + "XT.dat", coords, k // grid_size, n // grid_size, grid_size, grid_size)
    W = MatrixManager.Read(matrix_path + "W.dat", coords, k // grid_size, m // grid_size, grid_size, grid_size)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nCreating communicators...")

    if rank == 0:
        print("Communicators created!\nRunning multiplication...")

    # multiplication
    X_W = Cannon.x_transpose_w_cannon(X, W, dist.get_rank(), device, None)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4
    ex_X_W = MatrixManager.Read(matrix_path + "XT_W.dat", coords, n // grid_size, m // grid_size, grid_size, grid_size)

    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)

def test_Cannon_XWT(device):
    # setup rank and groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    grid_size = int(torch.sqrt(torch.tensor(world_size)))
    grid = (grid_size, grid_size)

    coords = ProcManager.GetCoords(rank, grid)

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
    X = MatrixManager.Read(matrix_path + "X.dat", coords, n // grid_size, k // grid_size, grid_size, grid_size)
    W = MatrixManager.Read(matrix_path + "WT.dat", coords, m // grid_size, k // grid_size, grid_size, grid_size)

    X = X.to(device=device)
    W = W.to(device=device)

    dist.barrier()
    if rank == 0:
        print("Matrices initialized!\nCreating communicators...")

    if rank == 0:
        print("Communicators created!\nRunning multiplication...")

    # multiplication
    X_W = Cannon.x_w_transpose_cannon(X, W, dist.get_rank(), device, None)

    dist.barrier()
    if rank == 0:
        print("Multiplication done!\nRunning final comparison...")
    # final comparison
    tol = 1e-4
    ex_X_W = MatrixManager.Read(matrix_path + "X_WT.dat", coords, n // grid_size, m // grid_size, grid_size, grid_size)

    ex_X_W = ex_X_W.to(device=device)

    res = MatrixManager.Compare(X_W, ex_X_W, tol)

    dist.barrier()
    if rank == 0:
        print("Comparison done!")

    assert(res)