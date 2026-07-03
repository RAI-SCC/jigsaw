"""Config for roll modules."""

from typing import Callable

import torch

from jigsaw import comm


class RollConfig:
    """Selection strategy for roll function."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
    ):
        self.comm_halo_1d_fn = self._comm_halo_1d_fn_selection(device)

    def _comm_halo_1d_fn_selection(self, device: torch.device) -> Callable:
        """
        Select comm call for halo 1d function.

        If cuda, comm will occur via all-to-all.
        If cpu, comm will occur via p2p.
        """
        if device.type == "cuda":
            return comm.comm_1d_alltoall
        elif device.type == "cpu":
            return comm.comm_1d_p2p
        else:
            raise ValueError(f"No halo 1d function possible for device {device.type}")
