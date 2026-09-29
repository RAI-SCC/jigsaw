"""Kernel selection for the distributed roll."""

from typing import Callable

import torch

from jigsaw import comm


class RollConfig:
    """Selection of the halo exchange of the distributed roll."""

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the configuration.

        Parameters
        ----------
        device : torch.device
            Device of the module.
        """
        self.comm_halo_1d_fn = self._comm_halo_1d_fn_selection(device)

    def _comm_halo_1d_fn_selection(self, device: torch.device) -> Callable:
        """Select the 1-D halo exchange: all-to-all on CUDA, point-to-point on CPU."""
        if device.type == "cuda":
            return comm.comm_1d_alltoall
        elif device.type == "cpu":
            return comm.comm_1d_p2p
        else:
            raise ValueError(f"No halo 1d function possible for device {device.type}")
