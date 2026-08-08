import time
import torch

import MatMul.src.Pytorch.Methods.MGR as MGR

class T_MGR:
    def __init__(self, grid, device, n_warm, n_iter, rank):
        self.grid = grid
        self.device = device

        self.n_warm = n_warm
        self.n_iter = n_iter

        self.rank = rank

    def X_W_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.X_W(X, W, self.device)
        torch.cuda.synchronize()

    def X_W(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()

        for _ in range(self.n_iter):
            MGR.X_W(X, W, self.device)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None

    def XT_W_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.XT_W(X, W, self.device)
        torch.cuda.synchronize()

    def XT_W(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()
        
        for _ in range(self.n_iter):
            MGR.XT_W(X, W, self.device)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None

    def X_WT_warmup(self, X, W):
        for _ in range(self.n_warm):
            MGR.X_WT(X, W, self.device)
        torch.cuda.synchronize()

    def X_WT(self, X, W):
        torch.cuda.synchronize()
        if self.rank == 0:
            s_t = time.perf_counter()
    
        for _ in range(self.n_iter):
            MGR.X_WT(X, W, self.device)
            
        torch.cuda.synchronize()

        if self.rank == 0:
            e_t = time.perf_counter()
            return (e_t - s_t) / self.n_iter
        return None