"""Tests for the tutorials in ``examples/`` (in-process, on the 4-rank test mesh).

Two kinds of test, both run under ``mpirun -n 4 pytest tests``:

* **Smoke** -- ``main()`` of every tutorial runs to completion. The tutorials that
  check themselves (``autoconvert``, ``reshard``, ``dataloader``) assert inside
  ``main()``, so running them *is* their parity check; the training tutorials must
  additionally bring their loss down.
* **Parity** -- the models the training tutorials define give the same output, loss
  and gradients as an undistributed version of the same model:

  ======================  ======================================================
  ``domain_parallel``     the same layers converted to plain ``nn.Conv2d`` etc.
  ``swin``                the same model on a mesh that does not split anything
  ``vit``                 a plain-PyTorch ViT carrying the same weights
  ======================  ======================================================

The launcher side (``torchrun --nproc-per-node 4 examples/x.py``) is covered by
``test_examples_launch.py``.
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import jigsaw
from jigsaw.modules.roll import DistributedRoll
from jigsaw.utils import set_all_seeds
from jigsaw.utils_aggregate import gather_1d_results, gather_results
from tests.parity import all_ranks_ok, no_tf32
from tests.parity import assert_close as _assert_close

pytestmark = [pytest.mark.dist_required, pytest.mark.dist(4)]

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

REL = 1e-4  # of the largest entry: fp32 noise here is ~1e-6, a wrong gradient >= 1e-2
# On CUDA the ring attention core runs in float16: ~5e-4 of the largest entry.
REL_FP16_ATTENTION = 5e-3


def _check(
    name: str,
    got: torch.Tensor,
    want: torch.Tensor,
    device: torch.device,
    group: dist.ProcessGroup,
    rel: float = REL,
) -> None:
    """Fail on every rank unless ``got`` matches ``want`` to ``rel`` of ``max|want|``.

    The tolerance scales with the tensor: a fixed absolute tolerance lets a badly wrong
    gradient through, since the gradients of a mean loss are tiny.
    """
    atol = rel * want.abs().max().item() + 1e-12
    _assert_close(name, got, want, atol, rel, device, group)


def _load(name: str) -> ModuleType:
    """Import ``examples/<name>.py`` without running it (it has a ``__main__`` guard)."""
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))  # the tutorials do ``from _common import``
    spec = importlib.util.spec_from_file_location(
        f"jigsaw_example_{name}", EXAMPLES / f"{name}.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example(
    monkeypatch: pytest.MonkeyPatch, device: torch.device
) -> Callable[[str], ModuleType]:
    """Load a tutorial whose ``setup()`` reuses the test session's process group."""

    def load(name: str) -> ModuleType:
        module = _load(name)
        monkeypatch.setattr(
            module, "setup", lambda: (device, dist.get_rank(), dist.get_world_size())
        )
        return module

    return load


def _agree(ok: bool, device: torch.device, message: str = "") -> None:
    """Fail on every rank if ``ok`` is false on any rank, instead of hanging the rest."""
    assert all_ranks_ok(ok, device, dist.group.WORLD), message


def _tile(
    x: torch.Tensor, mesh: dict, dims_groups: list[tuple[int, str]]
) -> torch.Tensor:
    """This rank's slice of ``x``: each ``(dim, mesh key)`` splits ``dim`` over that axis."""
    for dim, key in dims_groups:
        group = mesh[key]
        size = x.shape[dim] // group.size()
        x = x.narrow(dim, group.rank() * size, size)
    return x.contiguous()


class _TorchRoll(nn.Module):
    """``torch.roll`` as a module, the undivided stand-in for ``DistributedRoll``."""

    def __init__(self, shifts: tuple[int, ...], dims: tuple[int, ...]) -> None:
        super().__init__()
        self.shifts, self.dims = shifts, dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.roll(x, self.shifts, self.dims)


@pytest.fixture(scope="module")
def replica_mesh(device: torch.device) -> dict:
    """One full replica per rank: a mesh that does not split anything."""
    return jigsaw.make_process_mesh(
        {"dp": 4, "sh": 1, "sw": 1, "col": 1},
        device_type=device.type,
        model_parallel_axes=("sh", "sw", "col"),
    )


class TestSmoke:
    """``main()`` of every tutorial runs, and the training ones learn."""

    @pytest.mark.parametrize("name", ["domain_parallel", "vit", "swin"])
    def test_training_reduces_loss(
        self, example: Callable[[str], ModuleType], device: torch.device, name: str
    ) -> None:
        torch.manual_seed(0)
        losses = example(name).main(steps=30)
        assert len(losses) == 30
        first, last = sum(losses[:3]) / 3, sum(losses[-3:]) / 3
        _agree(
            all(torch.isfinite(torch.tensor(losses)).tolist()),
            device,
            "non-finite loss",
        )
        _agree(last < first, device, f"loss did not go down: {first:.4f} -> {last:.4f}")

    def test_autoconvert(self, example: Callable[[str], ModuleType]) -> None:
        example("autoconvert").main()  # asserts forward, backward and merge-back parity

    def test_reshard(self, example: Callable[[str], ModuleType]) -> None:
        example("reshard").main(steps=20)  # asserts plain-PyTorch and resharded parity

    def test_dataloader(self, example: Callable[[str], ModuleType]) -> None:
        # num_workers=0: no fork()ed workers inside an MPI process.
        example("dataloader").main(num_workers=0)


class TestParity:
    """The models of the tutorials against undistributed references."""

    def test_domain_parallel_matches_plain_pytorch(
        self,
        example: Callable[[str], ModuleType],
        spatial_mesh_2x2_hw: dict,
        device: torch.device,
    ) -> None:
        mesh = spatial_mesh_2x2_hw
        module = example("domain_parallel")
        torch.manual_seed(0)
        model = module.build_model(mesh, device)
        # The weights are replicated, so merging them back gives the full layers.
        plain = nn.Sequential(
            *[
                m.convert_to_torch(mesh, row_kw="sh")
                if hasattr(m, "convert_to_torch")
                else m
                for m in model
            ]
        )

        torch.manual_seed(1)
        x = torch.randn(2, 1, 64, 64, device=device)
        y = torch.randn(2, 1, 64, 64, device=device)
        tiles = [(-2, "sh"), (-1, "sw")]

        with no_tf32():
            out = model(_tile(x, mesh, tiles))
            loss_fn = jigsaw.loss.DistributedMSELoss(
                reduce_groups=(mesh["sh"], mesh["sw"])
            )
            loss = loss_fn(out, _tile(y, mesh, tiles))
            loss.backward()

            ref_out = plain(x)
            ref_loss = F.mse_loss(ref_out, y)
            ref_loss.backward()

        group = mesh["model"]
        full = gather_results(out.detach(), mesh, pm_key_row="sh", pm_key_col="sw")
        _check("output", full, ref_out.detach(), device, group)
        _check("loss", loss.detach(), ref_loss.detach(), device, group)
        for i, (layer, ref) in enumerate(zip(model, plain)):
            if hasattr(layer, "convert_to_torch"):
                for p in ("weight", "bias"):
                    got, want = getattr(layer, p).grad, getattr(ref, p).grad
                    _check(f"layer {i} {p}.grad", got, want, device, group)

    def test_swin_matches_undivided_image(
        self,
        example: Callable[[str], ModuleType],
        spatial_mesh_2x2_hw: dict,
        replica_mesh: dict,
        device: torch.device,
    ) -> None:
        """Tiles + ``DistributedRoll`` + local window attention == one big image.

        Every weight of the Swin tutorial is replicated, so the same model on a mesh
        with a single tile per rank (a plain, undivided model) is the reference. This
        covers the shifted-window mask, which depends on where a tile sits. The one
        layer swapped out in the reference is ``DistributedRoll``, for ``torch.roll``.
        """
        mesh = spatial_mesh_2x2_hw
        module = example("swin")
        torch.manual_seed(0)
        tiled = module.Swin(mesh, device)
        torch.manual_seed(0)
        whole = module.Swin(replica_mesh, device)
        # The same weights: the tiled model's are rank 0's, on every rank.
        for p, q in zip(whole.parameters(), tiled.parameters(), strict=True):
            p.data.copy_(q.data)
        for parent in whole.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, DistributedRoll):
                    setattr(parent, name, _TorchRoll(child.shifts, child.dims))

        torch.manual_seed(1)
        x = torch.randn(2, 1, module.H, module.W, device=device)
        y = torch.randn(2, 1, module.H, module.W, device=device)
        tiles = [(-2, "sh"), (-1, "sw")]

        with no_tf32():
            out = tiled(_tile(x, mesh, tiles))
            loss_fn = jigsaw.loss.DistributedMSELoss(
                reduce_groups=(mesh["sh"], mesh["sw"])
            )
            loss = loss_fn(out, _tile(y, mesh, tiles))
            loss.backward()

            ref_out = whole(x)
            ref_loss = F.mse_loss(ref_out, y)
            ref_loss.backward()

        group = mesh["model"]
        full = gather_results(out.detach(), mesh, pm_key_row="sh", pm_key_col="sw")
        _check("output", full, ref_out.detach(), device, group)
        _check("loss", loss.detach(), ref_loss.detach(), device, group)
        for (name, p), q in zip(
            tiled.named_parameters(), whole.parameters(), strict=True
        ):
            _check(f"{name}.grad", p.grad, q.grad, device, group)

    def test_vit_matches_plain_pytorch(
        self,
        example: Callable[[str], ModuleType],
        process_mesh_dict_2x2: dict,
        device: torch.device,
    ) -> None:
        """The ViT tutorial, with weights and data sharded, against plain PyTorch.

        The reference is the same architecture written with ``nn.Conv2d``,
        ``nn.LayerNorm``, ``nn.Linear`` and ``F.scaled_dot_product_attention``, holding
        the *merged* weights of the distributed model. This checks the whole forward
        pass -- including the global position embedding and the ring attention, whose
        result must not depend on how the tokens are split -- and, on the backward
        pass, the gradients of the input and of parameters of three different kinds
        (channel-sharded, sharded in both axes, attention).
        """
        mesh = process_mesh_dict_2x2
        row, col = mesh["row"], mesh["col"]
        module = example("vit")
        d, heads = module.DIM, module.HEADS
        h_loc = heads // col.size()

        set_all_seeds(mesh["model"].rank())  # as in the tutorial
        model = module.ViT(mesh, device)
        # Non-trivial layer-norm parameters; equal over "row", which replicates them.
        g = torch.Generator().manual_seed(100 + col.rank())
        with torch.no_grad():
            for m in model.modules():
                if isinstance(m, jigsaw.normalization.DistributedLayerNorm):
                    m.weight.copy_(1 + 0.2 * torch.randn(m.weight.shape, generator=g))
                    m.bias.copy_(0.2 * torch.randn(m.bias.shape, generator=g))

        # --- merge the distributed model into plain PyTorch ---------------------------
        def merged_norm(norm: nn.Module) -> nn.LayerNorm:
            plain = nn.LayerNorm(d, device=device)
            for name in ("weight", "bias"):
                full = gather_1d_results(
                    getattr(norm, name).data, mesh, dim=0, pm_key="col"
                )
                getattr(plain, name).data.copy_(full)
            return plain

        def qkv_to_local(w: torch.Tensor) -> torch.Tensor:
            """Plain fused ``(3 * E, in)`` weight -> this rank's ``(3 * E, in / col)``.

            The distributed attention stores, block by block, the ``(3, heads / col, hd)``
            rows of the heads that live on each column, with the inputs split over "col".
            """
            w = w.reshape(3, col.size(), h_loc, d // heads, d)
            w = w.permute(1, 0, 2, 3, 4).reshape(3 * d, d)
            return _tile(w, mesh, [(1, "col")])

        def merged_qkv(local: torch.Tensor) -> torch.Tensor:
            """Inverse of :func:`qkv_to_local`."""
            w = gather_1d_results(local, mesh, dim=1, pm_key="col")  # (3 * E, in)
            w = w.reshape(col.size(), 3, h_loc, d // heads, d)
            return w.permute(1, 0, 2, 3, 4).reshape(3 * d, d)

        class PlainAttention(nn.Module):
            def __init__(self, attn: nn.Module) -> None:
                super().__init__()
                self.qkv = nn.Linear(d, 3 * d, bias=False, device=device)
                self.head_concat = nn.Linear(d, d, device=device)
                self.qkv.weight.data.copy_(merged_qkv(attn.qkv.weight.data))
                # (out, E / col) and (out / col,): replicated over "row", split over "col"
                w = gather_1d_results(
                    attn.head_concat.weight.data, mesh, dim=1, pm_key="col"
                )
                b = gather_1d_results(
                    attn.head_concat.bias.data, mesh, dim=0, pm_key="col"
                )
                self.head_concat.weight.data.copy_(w)
                self.head_concat.bias.data.copy_(b)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                b, n, _ = x.shape
                q, k, v = (
                    self.qkv(x).view(b, n, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
                )
                out = F.scaled_dot_product_attention(q, k, v)
                return self.head_concat(out.transpose(1, 2).reshape(b, n, d))

        class PlainBlock(nn.Module):
            def __init__(self, block: nn.Module) -> None:
                super().__init__()
                self.norm1, self.norm2 = (
                    merged_norm(block.norm1),
                    merged_norm(block.norm2),
                )
                self.attn = PlainAttention(block.attn)
                self.mlp = nn.Sequential(
                    block.mlp[0].convert_to_torch(mesh),
                    nn.GELU(),
                    block.mlp[2].convert_to_torch(mesh),
                )

            def forward(self, z: torch.Tensor) -> torch.Tensor:
                z = z + self.attn(self.norm1(z))
                return z + self.mlp(self.norm2(z))

        class PlainViT(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = model.embed.convert_to_torch(mesh)
                self.blocks = nn.ModuleList([PlainBlock(b) for b in model.blocks])
                self.norm = merged_norm(model.norm)
                self.recover = model.recover.convert_to_torch(mesh)
                hp, wp = module.H // module.PATCH, module.W // module.PATCH
                pos = module.sincos_pos_embed(d, torch.arange(hp), torch.arange(wp))
                self.register_buffer("pos_embed", pos.to(device))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                z = self.embed(x)  # (B, E, Hp, Wp)
                b, e, hp, wp = z.shape
                z = (z.permute(0, 2, 3, 1) + self.pos_embed).reshape(b, hp * wp, e)
                for block in self.blocks:
                    z = block(z)
                z = self.norm(z).reshape(b, hp, wp, e).permute(0, 3, 1, 2)
                return x + self.recover(z)

        plain = PlainViT()

        # --- forward and backward on the same data ------------------------------------
        torch.manual_seed(1)  # the same on every rank
        x = torch.randn(2, module.IN_CHANNELS, module.H, module.W, device=device)
        y = torch.randn_like(x)
        tiles = [(1, "col"), (-1, "row")]  # channels over "col", width over "row"
        x_local = _tile(x, mesh, tiles).requires_grad_(True)
        x_full = x.clone().requires_grad_(True)

        with no_tf32():
            out = model(x_local)
            loss_fn = jigsaw.loss.DistributedMSELoss(reduce_groups=(row, col))
            loss = loss_fn(out, _tile(y, mesh, tiles))
            loss.backward()

            ref_out = plain(x_full)
            ref_loss = F.mse_loss(ref_out, y)
            ref_loss.backward()

        group = mesh["model"]
        full = gather_1d_results(out.detach(), mesh, dim=1, pm_key="col")
        full = gather_1d_results(full, mesh, dim=-1, pm_key="row")

        rel = REL_FP16_ATTENTION if device.type == "cuda" else REL

        def check(name: str, got: torch.Tensor, want: torch.Tensor) -> None:
            _check(name, got, want, device, group, rel)

        check("output", full, ref_out.detach())
        check("loss", loss.detach(), ref_loss.detach())
        check("x.grad", x_local.grad, _tile(x_full.grad, mesh, tiles))
        # Layer norms: sharded over "col", replicated over "row". The first norm of a
        # block sits in front of the attention, so its gradient is the one of the ring
        # attention's backward pass.
        check(
            "norm.weight.grad",
            model.norm.weight.grad,
            plain.norm.weight.grad.chunk(col.size())[col.rank()],
        )
        check(
            "norm1.weight.grad",
            model.blocks[1].norm1.weight.grad,
            plain.blocks[1].norm1.weight.grad.chunk(col.size())[col.rank()],
        )
        # An MLP weight, (4E, E): rows over "row", columns over "col".
        check(
            "mlp[0].weight.grad",
            model.blocks[0].mlp[0].weight.grad,
            _tile(plain.blocks[0].mlp[0].weight.grad, mesh, [(0, "row"), (1, "col")]),
        )
        # The attention's fused QKV, in the distributed layout.
        check(
            "attn.qkv.weight.grad",
            model.blocks[1].attn.qkv.weight.grad,
            qkv_to_local(plain.blocks[1].attn.qkv.weight.grad),
        )
