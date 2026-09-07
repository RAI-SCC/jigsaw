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
