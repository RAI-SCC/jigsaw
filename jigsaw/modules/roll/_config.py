"""Kernel selection for the distributed roll."""

import torch

from jigsaw._core.comm import CommOp, resolve_comm_op, select_comm_1d


class RollConfig:
    """Selection of the halo exchange of the distributed roll.

    Parameters
    ----------
    device : torch.device
        Device of the module.
    comm : CommOp or str
        Transport of the halo exchange. ``"auto"`` picks all-to-all on CUDA
        and ``batch_isend_irecv`` on CPU.
    """

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        comm: CommOp | str = CommOp.AUTO,
    ):
        """Initialize the configuration."""
        self.comm = resolve_comm_op(comm, device)
        self.comm_halo_1d_fn = select_comm_1d(self.comm)
