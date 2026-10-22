"""convert_to_jigsaw flags modules it leaves as plain torch.

None of these models contain a converted module, so no process groups are
needed and everything runs single-process.
"""

import warnings

import pytest
import torch

from jigsaw.convert import convert_to_jigsaw

DEVICE = torch.device("cpu")


class _OwnsParam(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.ones(4))
        self.act = torch.nn.ReLU()


class _Unknown(torch.nn.Module):
    """A leaf in neither ``shard_safe_modules`` nor ``shard_unsafe_modules``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def test_safe_modules_do_not_warn() -> None:
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.GELU())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert convert_to_jigsaw(model, {}, DEVICE) is model


def test_unknown_leaf_warns_with_qualified_name() -> None:
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sequential(_Unknown()))
    with pytest.warns(UserWarning, match=r"1\.0: unrecognized leaf module _Unknown"):
        convert_to_jigsaw(model, {}, DEVICE)


def test_direct_parameters_flagged() -> None:
    with pytest.warns(UserWarning, match=r"<root>: _OwnsParam owns parameters"):
        convert_to_jigsaw(_OwnsParam(), {}, DEVICE)


def test_raise_leaves_model_unchanged() -> None:
    model = torch.nn.Sequential(_Unknown())
    before = list(model.named_modules())
    with pytest.raises(ValueError, match="0: unrecognized leaf module _Unknown"):
        convert_to_jigsaw(model, {}, DEVICE, on_unrecognized="raise")
    assert list(model.named_modules()) == before


def test_ignore_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert_to_jigsaw(_Unknown(), {}, DEVICE, on_unrecognized="ignore")


def test_layer_norm_opt_out_not_flagged() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert_to_jigsaw(torch.nn.LayerNorm(4), {}, DEVICE, convert_layer_norm=False)


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError, match="on_unrecognized"):
        convert_to_jigsaw(torch.nn.ReLU(), {}, DEVICE, on_unrecognized="loud")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "module",
    [
        torch.nn.LogSoftmax(dim=-1),
        torch.nn.BatchNorm1d(4),
        torch.nn.MaxPool2d(2),
        torch.nn.ZeroPad2d(1),
        torch.nn.Flatten(),
        torch.nn.Embedding(8, 4),
        torch.nn.CrossEntropyLoss(),
        torch.nn.LSTM(4, 4),
    ],
    ids=lambda m: type(m).__name__,
)
def test_unsafe_module_raises(module: torch.nn.Module) -> None:
    model = torch.nn.Sequential(torch.nn.ReLU(), module)
    before = list(model.named_modules())
    with pytest.raises(
        ValueError,
        match=rf"non_safe=True.*\n  1: non-elementwise module {type(module).__name__}",
    ):
        convert_to_jigsaw(model, {}, DEVICE)
    assert list(model.named_modules()) == before


def test_unsafe_raises_even_when_ignoring_unrecognized() -> None:
    with pytest.raises(ValueError, match="non-elementwise module RMSNorm"):
        convert_to_jigsaw(torch.nn.RMSNorm(4), {}, DEVICE, on_unrecognized="ignore")


def test_unsafe_container_flagged_without_descending() -> None:
    layer = torch.nn.TransformerEncoderLayer(8, 2, dim_feedforward=16)
    with pytest.raises(ValueError) as excinfo:
        convert_to_jigsaw(torch.nn.Sequential(layer), {}, DEVICE)
    lines = str(excinfo.value).splitlines()[1:]
    assert lines == ["  0: non-elementwise module TransformerEncoderLayer"]


def test_non_safe_downgrades_to_warning() -> None:
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    with pytest.warns(UserWarning, match="0: non-elementwise module BatchNorm1d"):
        assert convert_to_jigsaw(model, {}, DEVICE, non_safe=True) is model


def test_non_safe_with_ignore_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert_to_jigsaw(
            torch.nn.AvgPool1d(2), {}, DEVICE, non_safe=True, on_unrecognized="ignore"
        )


def test_non_safe_with_raise_still_raises() -> None:
    with pytest.raises(ValueError, match="0: non-elementwise module GLU"):
        convert_to_jigsaw(
            torch.nn.Sequential(torch.nn.GLU()),
            {},
            DEVICE,
            non_safe=True,
            on_unrecognized="raise",
        )
