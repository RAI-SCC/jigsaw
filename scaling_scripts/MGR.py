import time
import torch

import jigsaw.modules.linear.mgr as MGR

class T_MGR:
    def __init__(self, grid, device, n_warm, n_iter, rank):
        self.grid = grid
        self.device = device

        self.n_warm = n_warm
        self.n_iter = n_iter

        self.rank = rank

    def X_W_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.x_w_mgr(X, W, self.rank, self.device, None)
        torch.cuda.synchronize()

    def X_W(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()

        for _ in range(self.n_iter):
            MGR.x_w_mgr(X, W, self.rank, self.device, None)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None

    def XT_W_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.x_transpose_w_mgr(X, W, self.rank, self.device, None)
        torch.cuda.synchronize()

    def XT_W(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()
        
        for _ in range(self.n_iter):
            MGR.x_transpose_w_mgr(X, W, self.rank, self.device, None)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None

    def X_WT_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.x_w_transpose_mgr(X, W, self.rank, self.device, None)
        torch.cuda.synchronize()

    def X_WT(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()
    
        for _ in range(self.n_iter):
            MGR.x_w_transpose_mgr(X, W, self.rank, self.device, None)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None