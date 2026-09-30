import pytest
import torch.distributed as dist

import jigsaw


class TestValidation:
    """Checks that hold at any world size -- run in the default session too."""

    def test_all_axes_form_the_model_group_by_default(self, device):
        world = dist.get_world_size()
        mesh = jigsaw.make_process_mesh({"col": world}, device_type=device.type)
        assert set(mesh) == {"col", "model"}
        assert mesh["col"].size() == world
        assert mesh["model"].size() == world

    def test_infers_single_negative_one_axis(self, device):
        mesh = jigsaw.make_process_mesh(
            {"dp": -1, "row": 1, "col": 1}, device_type=device.type
        )
        assert mesh["dp"].size() == dist.get_world_size()

    def test_rejects_mesh_that_does_not_cover_the_world(self):
        with pytest.raises(ValueError, match=r"needs exactly \d+ ranks"):
            jigsaw.make_process_mesh({"row": dist.get_world_size() + 1})

    def test_rejects_multiple_negative_one_axes(self):
        with pytest.raises(ValueError, match="at most one axis may be -1"):
            jigsaw.make_process_mesh({"a": -1, "b": -1})

    def test_rejects_non_trailing_model_axes(self):
        with pytest.raises(ValueError, match="trailing axes"):
            jigsaw.make_process_mesh({"row": 1, "col": 1}, model_parallel_axes=("row",))

    def test_rejects_model_key_colliding_with_an_axis(self):
        with pytest.raises(ValueError, match="collides with an axis"):
            jigsaw.make_process_mesh({"model": 1})

    def test_rejects_empty_axes(self):
        with pytest.raises(ValueError, match="at least one mesh dimension"):
            jigsaw.make_process_mesh({})

    def test_rejects_uninferable_axis(self):
        # A zero-sized fixed axis makes the product 0, so nothing can be inferred.
        with pytest.raises(ValueError, match="cannot infer -1 axis"):
            jigsaw.make_process_mesh({"a": -1, "b": 0})

    def test_rejects_world_not_divisible_by_fixed_axes(self):
        world = dist.get_world_size()
        with pytest.raises(ValueError, match="cannot infer -1 axis"):
            jigsaw.make_process_mesh({"a": -1, "b": world + 1})

    def test_rejects_non_positive_sizes(self):
        with pytest.raises(ValueError, match="every axis size must be >= 1"):
            jigsaw.make_process_mesh({"a": 0})
        with pytest.raises(ValueError, match="every axis size must be >= 1"):
            jigsaw.make_process_mesh({"a": -2})

    def test_rejects_empty_model_parallel_axes(self):
        with pytest.raises(ValueError, match="must be non-empty"):
            jigsaw.make_process_mesh({"a": -1}, model_parallel_axes=())

    def test_rejects_unknown_model_parallel_axes(self):
        with pytest.raises(ValueError, match="are not in axes"):
            jigsaw.make_process_mesh({"a": -1}, model_parallel_axes=("nope",))

    def test_requires_an_initialised_process_group(self, monkeypatch):
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        with pytest.raises(RuntimeError, match="initialised default process group"):
            jigsaw.make_process_mesh({"a": 1})

    def test_device_type_defaults_to_available_accelerator(self, monkeypatch):
        import torch

        from jigsaw import mesh

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert mesh._resolve_device_type(None) == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert mesh._resolve_device_type(None) == "cuda"
        assert mesh._resolve_device_type("cpu") == "cpu"


@pytest.mark.dist(4)
class TestFourRanks:
    def test_fully_model_parallel_2x2(self, device):
        mesh = jigsaw.make_process_mesh({"row": 2, "col": 2}, device_type=device.type)
        assert mesh["row"].size() == 2
        assert mesh["col"].size() == 2
        assert mesh["model"].size() == 4

    def test_replica_axis_is_kept_but_excluded_from_model(self, device):
        mesh = jigsaw.make_process_mesh(
            {"dp": 2, "row": 1, "col": 2},
            device_type=device.type,
            model_parallel_axes=("row", "col"),
        )
        assert mesh["dp"].size() == 2
        assert mesh["model"].size() == 2

    def test_spatial_axes_flatten_into_the_model_group(self, device):
        mesh = jigsaw.make_process_mesh(
            {"sh": 2, "sw": 2, "col": 1}, device_type=device.type
        )
        assert mesh["sh"].size() == 2
        assert mesh["model"].size() == 4
