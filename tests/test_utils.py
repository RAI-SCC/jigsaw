"""Unit tests for the checkpoint, seeding and precision helpers in jigsaw.utils."""

import os

import torch

from jigsaw import utils


def _model_and_optimizer():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    return model, optimizer


def test_set_all_seeds_is_reproducible_and_exports_seed(monkeypatch):
    monkeypatch.delenv("PL_GLOBAL_SEED", raising=False)
    utils.set_all_seeds(7)
    first = torch.randn(4)
    utils.set_all_seeds(7)
    assert torch.equal(first, torch.randn(4))
    assert os.environ["PL_GLOBAL_SEED"] == "7"
    monkeypatch.delenv("PL_GLOBAL_SEED")


def test_count_parameters_skips_frozen():
    model = torch.nn.Linear(3, 2)
    assert utils.count_parameters(model) == 3 * 2 + 2
    model.bias.requires_grad_(False)
    assert utils.count_parameters(model) == 3 * 2


def test_setup_precision_toggles_tf32():
    matmul, cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        utils.setup_precision(True)
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
        utils.setup_precision(False)
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


class TestCheckpointRoundTrip:
    def test_save_then_load_restores_state_and_advances_epoch(self, tmp_path):
        model, optimizer = _model_and_optimizer()
        model(torch.randn(5, 3)).sum().backward()
        optimizer.step()
        utils.save_model(model, optimizer, None, str(tmp_path), rank=0, epoch=4)

        fresh_model, fresh_optimizer = _model_and_optimizer()
        loaded_model, loaded_optimizer, epoch = utils.load_model(
            fresh_model, fresh_optimizer, str(tmp_path / "model_0.pt")
        )

        assert epoch == 5
        for saved, restored in zip(
            model.state_dict().values(), loaded_model.state_dict().values()
        ):
            assert torch.equal(saved, restored)
        assert loaded_optimizer.state_dict()["state"]  # momentum buffers came back

    def test_model_parallel_selects_file_by_rank_modulo_parallelism(self, tmp_path):
        model, optimizer = _model_and_optimizer()
        with torch.no_grad():
            model.weight.fill_(3.0)
        utils.save_model(model, optimizer, None, str(tmp_path), rank=1, epoch=0)

        fresh_model, fresh_optimizer = _model_and_optimizer()
        loaded, _, epoch = utils.load_model(
            fresh_model,
            fresh_optimizer,
            str(tmp_path),
            rank=5,
            model_parallel=True,
            parallelism=4,
        )
        assert epoch == 1
        assert torch.all(loaded.weight == 3.0)

    def test_custom_model_name_prefixes_the_file(self, tmp_path):
        model, optimizer = _model_and_optimizer()
        utils.save_model(
            model, optimizer, None, str(tmp_path), rank=2, epoch=0, model_name="enc"
        )
        assert (tmp_path / "enc_2.pt").is_file()

    def test_missing_checkpoint_starts_from_scratch(self, tmp_path):
        model, optimizer = _model_and_optimizer()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        loaded, _, epoch = utils.load_model(
            model, optimizer, str(tmp_path / "does_not_exist.pt")
        )
        assert epoch == 0
        for key, value in loaded.state_dict().items():
            assert torch.equal(before[key], value)
