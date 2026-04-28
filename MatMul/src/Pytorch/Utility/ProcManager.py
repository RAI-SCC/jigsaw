import torch.distributed as dist

def GetCoords(rank, grid_shape):
    """
    Translates a flat rank into grid (row, col) coordinates.
    grid_shape: tuple (num_rows, num_cols)
    """
    num_rows, num_cols = grid_shape
    r = rank // num_cols
    c = rank % num_cols
    return (r, c)

def GetRank(coords, grid_shape):
    """
    Translates (row, col) coordinates back into a flat rank.
    grid_shape: tuple (num_rows, num_cols)
    """
    num_rows, num_cols = grid_shape
    r, c = coords
    # Check if the coordinates are within valid bounds
    if not (0 <= r < num_rows and 0 <= c < num_cols):
        raise ValueError(f"Coordinates {coords} out of grid bounds {grid_shape}")
    return r * num_cols + c