# Mixed precision (autocast)

Wrap the forward in `torch.autocast` and run `backward()` outside it, as with plain PyTorch:

```python
with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
    y = model(x)
loss = loss_fn(y, target)
loss.backward()  # outside the autocast region
```

_jigsaw_'s layers are custom `torch.autograd.Function`s, which `torch.autocast` does **not**
descend into automatically, so each opts in by hand. Behavior:

- **Matmul family**: `DistributedLinear`, attention (plain + ring), `DistributedConv*` /
  `DistributedConvTranspose*`: the forward runs in the autocast dtype and the output carries
  it (bias included); gradients are reconciled to the **fp32 (parameter) dtype** in backward,
  i.e. you keep fp32 master gradients for free.
- **`DistributedLayerNorm` (with or without `elementwise_affine`) / `DistributedGroupNorm` /
  `DistributedSoftmax`** follow stock-AMP "fp32 list" semantics: computed and emitted in fp32 even when fed bf16/fp16.
- **Attention core (`sdpa` / `math` / `te`)**: under autocast it uses the autocast dtype
  (so `autocast(bfloat16)` stays bf16 and `autocast(float16)` fp16). With no autocast it uses
  the input dtype, except that fp32 and fp64 inputs on CUDA run the core in bf16: the flash
  kernel rejects fp32, and fp16 would underflow small gradients without a loss scaler. On CPU fp32
  stays fp32 (SDPA math backend). A float `attn_mask` is cast to the core dtype for you.

Only `float16` and `bfloat16` are autocast dtypes. `float64` params work for the matmul,
conv and norm layers but not attention (it force-casts q/k/v). FP8 is not supported:
TransformerEngine's `fp8_autocast` has no effect on _jigsaw_'s layers, including the
`te` attention backend, which runs non-FP8 kernels only.
