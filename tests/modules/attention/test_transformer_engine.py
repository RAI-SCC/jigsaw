"""TransformerEngine attention path (container-only).

These are skipped automatically unless ``transformer_engine`` is importable
(the ``te_required`` marker, see tests/conftest.py) and a GPU is present.
Run them inside the container: ``JIGSAW_SITE=<site> scripts/run.sh pytest -m te_required``.
"""

import pytest
import torch

pytestmark = [pytest.mark.te_required, pytest.mark.gpu_required]


def test_te_available_flag() -> None:
    """The import guard in attention.py flips when TE is present."""
    from jigsaw.modules.attention.attention import JIGSAW_TE_AVAILABLE

    assert JIGSAW_TE_AVAILABLE is True


def test_transformer_engine_attention_forward_backward(device: torch.device) -> None:
    """``TransformerEngineAttention`` runs fwd/bwd and matches SDPA within bf16 tol."""
    import transformer_engine.pytorch as te

    from jigsaw.modules.attention.attention import TransformerEngineAttention

    batch, seq, heads, head_dim = 2, 16, 4, 32
    attn = te.DotProductAttention(
        num_attention_heads=heads,
        kv_channels=head_dim,
        num_gqa_groups=heads,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    )

    torch.manual_seed(0)
    # bshd layout, as the module feeds it.
    q, k, v = (
        torch.randn(batch, seq, heads, head_dim, device=device, dtype=torch.bfloat16)
        for _ in range(3)
    )
    for t in (q, k, v):
        t.requires_grad_(True)

    # te.DotProductAttention(qkv_format="bshd") returns (b, s, h*d).
    out = TransformerEngineAttention.apply(q, k, v, attn)
    assert out.shape == (batch, seq, heads * head_dim)

    out.sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.detach().transpose(1, 2),
        k.detach().transpose(1, 2),
        v.detach().transpose(1, 2),
        is_causal=False,
    ).transpose(1, 2).reshape(batch, seq, heads * head_dim)
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
