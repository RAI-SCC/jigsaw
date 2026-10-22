# Attention and sequence parallelism

`DistributedAttention` has two independent switches: the **attention core**
(`attn_backend`) and the **sequence-parallel scheme** (`ring` or `ulysses`).

## Attention core

`attn_backend` selects the kernel that computes attention on each rank:

- `sdpa` (default): PyTorch's `scaled_dot_product_attention`; flash kernels on CUDA,
  the math kernel on CPU.
- `math`: an unfused reference implementation.
- `te`: TransformerEngine's fused cuDNN core on Hopper and newer, the `flash_attn` core
  on older GPUs and on ROCm. Needs `transformer_engine` or `flash_attn`, e.g. inside an
  NGC PyTorch container (see [containers/README.md](../containers/README.md)); with
  neither importable it warns and falls back to `sdpa`. `JIGSAW_TE_CORE=cudnn|flash`
  overrides the choice of core (`cudnn` on ROCm is untested).

## Sequence parallelism: ring vs. Ulysses

Both schemes parallelize attention over `spatial_group` and work with any
`attn_backend`; they are mutually exclusive.

- `ring=True`: chunks circulate around `spatial_group` (the input `x`, or K/V, see
  below), with a per-step online-softmax merge. Works for any head count.
- `ulysses=True`: a single all-to-all trades the local sequence shard for a
  head shard, runs one dense attention over the full sequence, then an
  all-to-all brings it back. Needs `heads // channel_group.size()` divisible
  by `spatial_group.size()`. Set `ulysses_gpus_per_node > 0` to run the
  *hierarchical* scheme instead: Ulysses within intra groups of that many spatial
  ranks, nested in an outer ring across them, so the all-to-all stays within an
  intra group. It needs `heads // channel_group.size()` divisible by
  `ulysses_gpus_per_node`, and at least two intra groups.

Ring attention splits the sequence over the mesh rows; the mesh columns split the
heads, with or without `ring`. With split heads (or `qk_norm`) the ring has to
circulate the projected keys and values instead of the input `x`, which is only a
slice of the features. Without a `config` this is chosen for you; an explicit
`RingAttentionConfig` must set `send_in_forward=SendForward.KV` and
`save_for_backward=SaveBackward.KV` (the default), or the module raises a
`ValueError`. [`examples/vit.py`](../examples/vit.py) shows the explicit version.

`RingAttentionConfig.send_op` picks how the ring exchanges chunks. The default,
`ALLTOALL`, works for any ring size. The `HYBRID_*` ops split the ring into intra
groups of `gpus_per_node` ranks (default 4) joined by an outer ring, and need the ring
size to be a multiple of `gpus_per_node`.

The ring and hierarchical Ulysses can overlap their K/V communication with the
attention core on a dedicated CUDA stream: `overlap_comm` on `RingAttentionConfig` for the plain
ring, `ulysses_inter_overlap` for the hierarchical scheme's outer ring.
CUDA only; inert elsewhere. See `RingAttentionConfig.__init__` and
`DistributedAttention.__init__` for the full parameter reference.
