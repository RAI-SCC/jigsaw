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


def test_safe_modules_do_not_warn() -> None:
    model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Dropout(), torch.nn.GELU())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert convert_to_jigsaw(model, {}, DEVICE) is model


def test_unknown_leaf_warns_with_qualified_name() -> None:
    model = torch.nn.Sequential(
        torch.nn.ReLU(), torch.nn.Sequential(torch.nn.BatchNorm1d(4))
    )
    with pytest.warns(UserWarning, match=r"1\.0: unrecognized leaf module BatchNorm1d"):
        convert_to_jigsaw(model, {}, DEVICE)


def test_direct_parameters_flagged() -> None:
    with pytest.warns(UserWarning, match=r"<root>: _OwnsParam owns parameters"):
        convert_to_jigsaw(_OwnsParam(), {}, DEVICE)


def test_raise_leaves_model_unchanged() -> None:
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4))
    before = list(model.named_modules())
    with pytest.raises(ValueError, match="0: unrecognized leaf module Embedding"):
        convert_to_jigsaw(model, {}, DEVICE, on_unrecognized="raise")
    assert list(model.named_modules()) == before


def test_ignore_is_silent() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert_to_jigsaw(torch.nn.BatchNorm1d(4), {}, DEVICE, on_unrecognized="ignore")


def test_layer_norm_opt_out_not_flagged() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        convert_to_jigsaw(torch.nn.LayerNorm(4), {}, DEVICE, convert_layer_norm=False)


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError, match="on_unrecognized"):
        convert_to_jigsaw(torch.nn.ReLU(), {}, DEVICE, on_unrecognized="loud")  # type: ignore[arg-type]
