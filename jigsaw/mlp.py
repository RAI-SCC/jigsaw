import math

import torch
import torch.distributed as dist
import torch.nn.init as init
from torch.nn import GELU, Dropout

from . import linear, linear_megatron


class DistributedMLP(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        dropout: float,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        model_group: dist.ProcessGroup,
        spatial_group: dist.ProcessGroup = None,
        channel_group: dist.ProcessGroup = None,
    ):
        """
        Initialize the MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Global input dimension of MLP.
        hidden_dim : int
            Global hidden dimension of MLP.
        out_dim : int
            Global output dimension of MLP.
        device : torch.device
            Device to perform computation on.
        local_rank : int
            Model local_rank.
        parallelism : int, optional
            Degree of parallelism (size of model_group).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        dropout : int, optional
            dropout rate. Default value is 1e-3.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.linear1 = linear.DistributedXWT(
            in_dim // self.channel_group.size(),
            hidden_dim // self.spatial_group.size(),
            bias,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
            synchronized_layer=False
        )
        self.linear2 = linear.DistributedXWT(
            hidden_dim // self.channel_group.size(),
            out_dim // self.spatial_group.size(),
            bias,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
            synchronized_layer=False
        )

        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)

    def forward(self, x):
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x