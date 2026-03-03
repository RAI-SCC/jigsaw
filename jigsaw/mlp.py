import math

import torch
import torch.distributed as dist
import torch.nn.init as init
from torch.nn import GELU, Dropout

from . import linear, linear_megatron
from .utils_init import average_weights

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
    ):
        """
        Initialize the MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Input dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        hidden_dim : int
            Hidden dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        out_dim : int
            Output dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        device : torch.device
            Device to perform computation on.
        local_rank : int
            Model local_rank.
        parallelism : int, optional
            Degree of parallelism (default is 4).
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

        match parallelism:
            case 2:
                in_dim, hidden_dim1, hidden_dim2, out_dim = (
                    in_dim // 2,
                    hidden_dim,
                    hidden_dim // 2,
                    out_dim,
                )
            case 4:
                in_dim, hidden_dim1, hidden_dim2, out_dim = (
                    in_dim // 2,
                    hidden_dim // 2,
                    hidden_dim // 2,
                    out_dim // 2,
                )
            case 8:
                in_dim, hidden_dim1, hidden_dim2, out_dim = (
                    in_dim // 2,
                    hidden_dim // 4,
                    hidden_dim // 2,
                    out_dim // 4,
                )

        self.linear1 = linear.DistributedXWT(
            in_dim,
            hidden_dim1,
            bias,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
        )
        self.linear2 = linear.DistributedXWT(
            hidden_dim2,
            out_dim,
            bias,
            local_rank,
            device,
            parallelism,
            model_group,
            spatial_group,
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


class DistributedMLPNICE(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        bias: bool,
        local_rank: int,
        device: torch.device,
        parallelism: int,
        spatial_group: dist.ProcessGroup = None,
        channel_group: dist.ProcessGroup = None,
    ):
        """
        Initialize the MLP module for NICE Norm.

        Parameters
        ----------
        in_dim : int
            Global input dimension of noise.
        hidden_dim : int
            Global hidden dimension of MLP.
        out_dim : int
            Global output dimension of MLP.
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        device : torch.device
            Device to perform computation on.
        parallelism: int
            Number of processes that model is distributed over.
        local_rank : int
            Model local_rank.
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.spatial_group = spatial_group
        self.channel_group = channel_group

        self.linear1 = linear.DistributedXWTNice(
            in_dim,
            hidden_dim // 2,
            bias,
            device,
            spatial_group=spatial_group,
            channel_group=channel_group,
        )
        self.linear2 = linear.DistributedXWTSync(
            hidden_dim // 2,
            out_dim,
            bias,
            local_rank,
            device,
            parallelism=parallelism,
            sync_group=spatial_group,
            model_group=channel_group,
        )

        self.activation = GELU()

    def forward(self, x):
        """
        Forward pass of MLP distributed across two or four processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, x_in, in_dim).
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        return x


class DistributedSynchronizedMLP(torch.nn.Module):
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
        model_group: torch.distributed.ProcessGroup,
        sync_group: torch.distributed.ProcessGroup,
        bayesian: bool = False,
    ):
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group

        if parallelism == 2:
            # in, hidden, out dim are global sizes for two-way model parallelism
            # required to maintain partitioning across final dimension

            # local X has shape [seq_length, in_dim//2]
            self.linear1 = linear.DistributedXWTSync(
                in_dim // 2,
                hidden_dim,
                bias,
                local_rank,
                device,
                parallelism,
                model_group,
                sync_group,
                bayesian=bayesian,
            ).to(device)

            self.linear2 = linear.DistributedXWTSync(
                hidden_dim // 2,
                out_dim,
                bias,
                local_rank,
                device,
                parallelism,
                model_group,
                sync_group,
                bayesian=bayesian,
            ).to(device)

        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.local_rank = local_rank

        average_weights(self, sync_group)

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


class DistributedXTWMLP(torch.nn.Module):
    """Distributed MLP layer where inputs are transposed."""

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        bias,
        dropout,
        local_rank,
        device,
        parallelism,
        model_group,
    ):
        """
        Initialize the transposed XTW MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Global dimension of MLP.
        hidden_dim : int
            Global hidden dimension of MLP.
        out_dim : int
            Global output dimension of MLP.
        device : torch.device
            Device to perform computation on.
        local_rank : int
            Local model rank.
        parallelism : int, optional
            The level of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        """
        super().__init__()
        self.local_rank = local_rank
        self.device = device
        self.parallelism = parallelism
        self.model_group = model_group

        match parallelism:
            case 2:
                in_dim, hidden_dim, out_dim = in_dim, hidden_dim // 2, out_dim
            case 4:
                in_dim, hidden_dim, out_dim = in_dim // 2, hidden_dim // 2, out_dim // 2
            case 8:
                in_dim, hidden_dim, out_dim = in_dim // 4, hidden_dim // 2, out_dim // 4

        self.linear1 = linear.DistributedXTW(
            in_dim, hidden_dim, bias, local_rank, device, parallelism, model_group
        )
        self.linear2 = linear.DistributedXWT(
            hidden_dim, out_dim, bias, local_rank, device, parallelism, model_group
        )

        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.local_rank = local_rank

    def forward(self, x):
        """
        Forward pass of transposed MLP distributed across n processes.

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


class SequentialMLP(torch.nn.Module):
    """Multi-Layer Perceptron (MLP) with one hidden layer."""

    def __init__(self, in_dim, hidden_dim, out_dim, bias=True, dropout=1e-3):
        """
        Initialize MLP.

        ----------
        in_dim : int
            Dimension of the input features.
        hidden_dim : int
            Dimension of the hidden layer.
        out_dim : int
            Dimension of the output layer.
        dropout : float, optional
            Dropout probability, by default 1e-3.
        """
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim, bias=bias)  # first dim 4
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim, bias=bias)  # first dim 20
        init.uniform_(
            self.linear1.weight, a=-1.0 / math.sqrt(in_dim), b=1.0 / math.sqrt(in_dim)
        )
        init.uniform_(
            self.linear2.weight,
            a=-1.0 / math.sqrt(hidden_dim),
            b=1.0 / math.sqrt(hidden_dim),
        )
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)

    def forward(self, x):
        """Forward pass of MLP."""
        x = self.linear1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)

        return x


class DistributedMLPMegatron(torch.nn.Module):
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
        model_group: torch.distributed.ProcessGroup,
    ):
        """
        Initialize the MLP module for either two- or four-way model parallelism.

        Parameters
        ----------
        in_dim : int
            Input dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        hidden_dim : int
            Hidden dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        out_dim : int
            Output dimension of MLP.
            Two-way model: global value.
            Four-way model: local value.
        device : torch.device
            Device to perform computation on.
        local_rank : int
            Local model rank.
        parallelism : int, optional
            Degree of parallelism (default is 4).
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

        self.linear1 = linear_megatron.FirstLinearLayer(
            in_dim,
            hidden_dim,
            parallelism,
            bias=bias,
            device=device,
            model_group=self.model_group,
        )
        self.linear2 = linear_megatron.SecondLinearLayer(
            hidden_dim,
            out_dim,
            parallelism,
            bias=bias,
            device=device,
            model_group=self.model_group,
        )

        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.local_rank = local_rank

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
