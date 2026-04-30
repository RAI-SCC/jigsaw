"""
Ring Attention — bidirectional, sequence-parallel, torch.distributed (NCCL).

Each rank holds a local slice of Q, K, V of shape (B, H, N//world_size, d).
K and V are rotated around the ring for `world_size` steps. Each step calls
F.scaled_dot_product_attention for the local chunk, recovers (m_j, l_j) for
the online softmax merge, and accumulates O.

Backward uses the analytic attention gradient with globally-saved L.
dK/dV rotate counter-clockwise (opposite to forward) back to their home rank.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist

from . import ring_attention_helpers as ring_attn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ring_send_recv(tensor, world_size, rank, group, empty, recv_buf):
    send_to = (rank + 1) % world_size
    recv_from = (rank - 1) % world_size
    output_list = [recv_buf if i == recv_from else empty for i in range(world_size)]
    input_list = [tensor if i == send_to else empty for i in range(world_size)]
    handle = dist.all_to_all(output_list, input_list, group=group, async_op=True)
    return recv_buf, handle

def _online_softmax_merge(O1, L1, O2, L2):
    L_new = torch.logaddexp(L1, L2)
    e1 = torch.exp(L1 - L_new)
    e2 = torch.exp(L2 - L_new)
    O_new = e1.unsqueeze(-1) * O1 + e2.unsqueeze(-1) * O2
    return O_new, L_new


# notes on things that partially worked

# native flash attention
# with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
#     O_jflash = torch.nn.functional.scaled_dot_product_attention(
#         Q, Kj, Vj, is_causal=False, dropout_p=0, scale=scale
#     )

class DistributedRingAttention(torch.nn.Module):
    """Ring attention."""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device: torch.device,
        spatial_group: dist.ProcessGroup = None
    ):
        super().__init__()
        
        self.world_size = spatial_group.get_size()
        self.ring_attention = RingAttention.apply
        
    def forward(self):
        attn = self.RingAttention()
        
        
class RingAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, X, W_qkv, scale, group, empty, zeros, rank, world_size):
        B, N_local, H, d = Q.shape
        world_size = dist.get_world_size(group)
        O = torch.zeros_like(Q)
        l = torch.full((B, Nlocal, H), float("-inf"), device=Q.device, dtype=Q.dtype)

        Xj = X.clone()

        KV_chunks = []
        Xj = X.clone()
        recv_buf = torch.zeros_like(X)

        # Kick off first send before any compute
        recv_buf, handle = _ring_send_recv(Xj, world_size, rank, group, empty, recv_buf)
        for step in range(4):
            QKVj = (
                (Xj @ W_qkv[d * h :, :].mT)
                .reshape(B, Nlocal, 2, H, d)
                .permute(2, 0, 1, 3, 4)
            )
            Kj = QKVj[0, :].contiguous()
            Vj = QKVj[1, :].contiguous()
            # Recompute KV from current Xj while communication is in flight

            KV_chunks.append([Kj, Vj])

            # private flash attention?
            O_j, l_j, *_ = torch.ops.aten._flash_attention_forward(
                Q,
                Kj,
                Vj,
                cum_seq_q=None,
                cum_seq_k=None,
                max_q=Nlocal,
                max_k=Nlocal,
                dropout_p=0.0,
                is_causal=False,
                scale=scale,
                return_debug_mask=False,
            )

            O, l = _online_softmax_merge(O, l, O_j, l_j.mT)

            if step < world_size - 1:
                handle.wait()
                Xj = recv_buf.clone()
                recv_buf2 = torch.zeros_like(X)  # fresh buffer for next recv
                recv_buf2, handle = _ring_send_recv(
                    Xj, world_size, rank, group, empty, recv_buf2
                )
                recv_buf = recv_buf2

        # L = m + torch.log(l)  # global log-sum-exp, shape (B, H, Nlocal)
        O = O.to(torch.bfloat16)
        ctx.save_for_backward(Q, O, l.mT, *[t for pair in KV_chunks for t in pair])
        ctx.W_qkv = W_qkv
        ctx.empty = empty
        ctx.scale = scale
        ctx.group = group
        ctx.world_size = world_size

        return O

    @staticmethod
    def backward(ctx, dO):
        scale = ctx.scale
        group = ctx.group
        world_size = ctx.world_size
        rank = dist.get_rank(group)

        Q, O, L = ctx.saved_tensors[:3]
        kv_flat = ctx.saved_tensors[3:]
        KV_chunks = [(kv_flat[2 * i], kv_flat[2 * i + 1]) for i in range(world_size)]

        B, Nlocal, H, d = Q.shape

        dQ = torch.zeros_like(Q, dtype=Q.dtype)

        # We need to rotate dK, dV counter-clockwise back to home ranks
        # At step s in forward, KV came from rank (rank - s) % world_size
        # So dKj/dVj must go to rank (rank - s) % world_size
        # We do this by rotating dKj/dVj one step counter-clockwise per step

        dK = torch.zeros_like(Q, dtype=Q.dtype)
        dV = torch.zeros_like(Q, dtype=Q.dtype)

        for step in range(world_size):
            # Xj = KV[step]
            # QKVj = (
            #     (Xj @ W_qkv[d * h :, :].mT)
            #     .reshape(B, Nlocal, 2, H, d)
            #     .permute(2, 0, 1, 3, 4)
            # )
            home_rank = (rank - step) % world_size
            Kj, Vj = KV_chunks[step]
            Kj = Kj.contiguous()
            Vj = Vj.contiguous()
            # L for this rank was saved as (B*H, Nlocal) — need full L not per-step
            # Use saved global L
            dQ_j, dKj, dVj = (
                torch.ops.aten._scaled_dot_product_flash_attention_backward(
                    dO,
                    Q,
                    Kj,
                    Vj,
                    O,
                    L,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=scale,
                    cum_seq_q=None,
                    cum_seq_k=None,
                    max_q=Nlocal,
                    max_k=Nlocal,
                    philox_seed=torch.zeros(1, dtype=torch.long, device=Q.device),
                    philox_offset=torch.zeros(1, dtype=torch.long, device=Q.device),
                )
            )
            dQ += dQ_j

            if home_rank == rank:
                dK += dKj
                dV += dVj

        return dQ, dK, dV, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ring_attention(
    Q,
    K,
    V,
    X,
    W_qkv,
    scale=None,
    group=None,
    empty=None,
    zeros=None,
    rank=0,
    world_size=4,
):
    """
    Q, K, V : (B, H, N//world_size, d)  — already sharded along sequence dim.
    scale   : defaults to 1/sqrt(d).
    group   : process group; defaults to the global group.
    """
    return RingAttentionFunction.apply(
        Q, K, V, X, W_qkv, scale, group, empty, zeros, rank, world_size
    )


# ---------------------------------------------------------------------------
# Multi-rank test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import time

    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_id = int(os.environ["SLURM_LOCALID"])
    device = f"cuda:{local_id}"

    torch.cuda.set_device(local_id)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, init_method="env://"
    )

    torch.manual_seed(rank)  # different data per rank
    B, h, N, d = 1, 16, world_size * 16382, 256
    print(f'total produce: {B*h*N*d}')
    Nlocal = N // world_size
    scale = d**-0.5

    # Each rank generates its own local slice
    torch.manual_seed(3)
    W_qkv = torch.randn(
        3 * d * h, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)
    torch.manual_seed(rank)
    X = torch.randn(
        B, Nlocal, d, device=device, requires_grad=True, dtype=torch.bfloat16
    ) / (d**0.5)

    QKV = X @ W_qkv.mT
    # QKV shape: [B, Nlocal, 3*d*h]
    Q_local, K_local, V_local = QKV.reshape(B, Nlocal, 3, h, d).permute(2, 0, 3, 1, 4)

    Q_local, K_local, V_local = (
        Q_local.contiguous(),
        K_local.contiguous(),
        V_local.contiguous(),
    )
    O = torch.zeros_like(Q_local)

    # Build consistent full tensors for reference via all-gather
    def all_gather_seq(t):
        chunks = [torch.empty_like(t) for _ in range(world_size)]
        dist.all_gather(chunks, t.detach().contiguous(), group=dist.group.WORLD)
        return torch.cat(chunks, dim=2)  # (B, H, N, d)

    Q_full = all_gather_seq(Q_local)
    K_full = all_gather_seq(K_local)
    V_full = all_gather_seq(V_local)

    sl = slice(rank * Nlocal, (rank + 1) * Nlocal)

    # Reference: full-sequence SDPA, slice out this rank's portion
    Q_ref = Q_full.detach().requires_grad_(True).transpose(1, 2)
    K_ref = K_full.detach().requires_grad_(True).transpose(1, 2)
    V_ref = V_full.detach().requires_grad_(True).transpose(1, 2)
    Q_ref, K_ref, V_ref = Q_ref.contiguous(), K_ref.contiguous(), V_ref.contiguous()
    # Shape of Q is now: [B, N, H, D]

    dO = torch.ones([B, h, Nlocal, d], dtype=torch.bfloat16, device=device)
    dO_full = all_gather_seq(dO).transpose(1, 2)
    
    dO = dO.transpose(1, 2).contiguous()

    # warmup
    for i in range(5):
        O_ref_full = torch.ops.aten._flash_attention_forward(
            Q_ref,
            K_ref,
            V_ref,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )

    dist.barrier()
    torch.cuda.synchronize()
    ref_start_time = time.perf_counter()

    n_reps = 1
    for _ in range(n_reps):
        O_ref_full, logsumexp, *_ = torch.ops.aten._flash_attention_forward(
            Q_ref,
            K_ref,
            V_ref,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            return_debug_mask=False,
        )

        dQ, dK, dV = torch.ops.aten._scaled_dot_product_flash_attention_backward(
            torch.ones(dO_full.shape, device=device, dtype=dO_full.dtype),  # grad_output: (B, N, H, d)
            Q_ref,  # query:       (B, N, H, d)
            K_ref,  # key:         (B, N, H, d)
            V_ref,  # value:       (B, N, H, d)
            O_ref_full,  # out:         (B, H, N, d)
            logsumexp,  # logsumexp:   (B*H, N) — from forward, reshape(B*H, N) before passing
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            cum_seq_q=None,
            cum_seq_k=None,
            max_q=Nlocal,
            max_k=Nlocal,
            philox_seed=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
            philox_offset=torch.zeros(1, dtype=torch.long, device=Q_ref.device),
        )
    torch.cuda.synchronize()
    dist.barrier()
    if rank == 0:
        ref_end_time = time.perf_counter()

    O_ref_local = O_ref_full[:, sl, :, :].detach()
    dQ_ref = dQ[:, sl, :, :].detach()
    dK_ref = dK[:, sl, :, :].detach()
    dV_ref = dV[:, sl, :, :].detach()
    # tiny_tensor = torch.zeros(4, device=X.device, dtype=X.dtype)
    empty = torch.empty(0, device=X.device, dtype=X.dtype)
    zeros = torch.zeros_like(X, device=X.device, dtype=X.dtype)
    scale = Q_local.shape[-1] ** -0.5
    # Ring attention warmup
    Q_local, K_local, V_local = (
        Q_local.transpose(1, 2),
        K_local.transpose(1, 2),
        V_local.transpose(1, 2),
    )
    for i in range(10):
        O_ring = ring_attention(
            Q_local,
            K_local,
            V_local,
            X,
            W_qkv,
            scale,
            None,
            empty,
            zeros,
            rank,
            world_size,
        )

    if rank == 0:
        print(
            f"[R{rank}/{dist.get_world_size(group=None)}] Q_ref shape {Q_ref.shape} Q_local shape {Q_local.shape}"
        )
    dist.barrier()
    torch.cuda.synchronize()
    ring_start_time = time.perf_counter()
    
    for _ in range(n_reps):
        O_ring = ring_attention(
            Q_local,
            K_local,
            V_local,
            X,
            W_qkv,
            scale,
            None,
            empty,
            zeros,
            rank,
            world_size,
        )
        Q_local.retain_grad()
        K_local.retain_grad()
        V_local.retain_grad()
        # O_ring.retain_grad()
        O_ring.sum().backward()

    torch.cuda.synchronize()
    dist.barrier()
    ring_end_time = time.perf_counter()

    atol = 1e-4
    print(
        f"[R{rank}/{dist.get_world_size(group=None)}] Forward  match: {torch.allclose(O_ring.detach(), O_ref_local, atol=atol)}"
    )
    print(
        f"[R{rank}/{dist.get_world_size(group=None)}] dQ       match: {torch.allclose(Q_local.grad, dQ_ref, atol=atol)}"
    )
    print(
        f"[R{rank}/{dist.get_world_size(group=None)}] dK       match: {torch.allclose(K_local.grad, dK_ref, atol=atol)}"
    )
    print(
        f"[R{rank}/{dist.get_world_size(group=None)}] dV       match: {torch.allclose(V_local.grad, dV_ref, atol=atol)}"
    )

    if rank == 0:
        print(
            f"[R{rank}/{dist.get_world_size(group=None)}] ref time: {(ref_end_time - ref_start_time) / n_reps}; ring time {(ring_end_time - ring_start_time) / n_reps}\n"
            f"[R{rank}/{dist.get_world_size(group=None)}] speedup: {(ref_end_time - ref_start_time) / (ring_end_time - ring_start_time)}"
        )

    dist.destroy_process_group()
