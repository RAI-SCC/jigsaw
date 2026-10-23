"""Kernel selection for the distributed roll."""

from collections.abc import Callable

import torch

from jigsaw._core import comm


class RollConfig:
    """Selection of the halo exchange of the distributed roll.

    Parameters
    ----------
    device : torch.device
        Device of the module.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize the configuration."""
        self.comm_halo_1d_fn = self._comm_halo_1d_fn_selection(device)

    def _comm_halo_1d_fn_selection(self, device: torch.device) -> Callable:
        """Select the 1-D halo exchange: all-to-all on CUDA, point-to-point on CPU.

        Raises
        ------
        ValueError
            If ``device`` is neither CUDA nor CPU.
        """
        if device.type == "cuda":
            return comm.comm_1d_alltoall
        elif device.type == "cpu":
            return comm.comm_1d_p2p
        else:
            raise ValueError(f"No halo 1d function possible for device {device.type}")
