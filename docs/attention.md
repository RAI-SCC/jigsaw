# Attention and sequence parallelism

`DistributedAttention` has two independent switches: the **attention core**
(`attn_backend`) and the **sequence-parallel scheme** (`ring` or `ulysses`).

## Attention core

`attn_backend` selects the kernel that computes attention on each rank:

- `sdpa` (default) — PyTorch's `scaled_dot_product_attention` (flash / math).
- `math` — an unfused reference implementation.
- `te` — TransformerEngine. Needs a working `transformer_engine`, e.g. inside an
  NGC PyTorch container (see [containers/README.md](../containers/README.md));
  otherwise it warns and falls back to `sdpa`. On ROCm it uses the `flash_attn`
  core (`JIGSAW_TE_CORE=cudnn` forces TE's fused core, untested).

## Sequence parallelism: ring vs. Ulysses

Both schemes parallelise attention over `spatial_group` and work with any
`attn_backend`; they are mutually exclusive.

- `ring=True` — K/V circulate around `spatial_group` in a P2P ring, with a
  per-step online-softmax merge. Works for any head count.
- `ulysses=True` — a single all-to-all trades the local sequence shard for a
  head shard, runs one dense attention over the full sequence, then an
  all-to-all brings it back. Needs `heads // channel_group.size()` divisible
  by `spatial_group.size()`. Set `ulysses_gpus_per_node > 0` to run the
  *hierarchical* scheme instead — Ulysses over an intra-node sub-group nested
  in a P2P ring over the inter-node one — which keeps the all-to-all on the
  fast intra-node fabric.

Ring attention splits the sequence over the mesh rows; the mesh columns split the
heads, with or without `ring`. With split heads (or `qk_norm`) the ring has to
circulate the projected keys and values instead of the input `x`, which is only a
slice of the features. Without a `config` this is chosen for you; an explicit
`RingAttentionConfig` must set `send_in_forward=SendForward.KV` and
`save_for_backward=SaveBackward.KV` (the default), or the module raises a
`ValueError`. [`examples/vit.py`](../examples/vit.py) shows the explicit version.

`RingAttentionConfig.send_op` picks how the ring exchanges chunks. The default,
`ALLTOALL`, works for any ring size. The `HYBRID_*` ops split the ring into intra- and
inter-node parts and need the ring size to be a multiple of `gpus_per_node`
(default 4).

Both schemes can overlap their K/V communication with the attention core on a
dedicated CUDA stream: `overlap_comm` on `RingAttentionConfig` for the plain
ring, `ulysses_inter_overlap` for the hierarchical scheme's inter-node ring.
CUDA only; inert elsewhere. See `RingAttentionConfig.__init__` and
`DistributedAttention.__init__` for the full parameter reference.
