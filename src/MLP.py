import torch
from torch.nn import GELU, Dropout
import Linear, Linear_Megatron


class DistributedMLP(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: torch.distributed.ProcessGroup,
        bias: bool = True,
        dropout: float = 1e-3
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
        rank : int
            Global rank.
        parallelism : int, optional
            Degree of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        dropout : int, optional
            dropout rate. Default value is 1e-3.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        if n_parallel == 2:
            # in, hidden, out dim are global sizes for two-way model parallelism
            # required to maintain partitioning across final dimension
            self.linear1 = Linear.DistributedXWT(
                in_dim//2, hidden_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXWT(
                hidden_dim//2, out_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 4:
            self.linear1 = Linear.DistributedXWT(
                in_dim//2, hidden_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXWT(
                hidden_dim//2, out_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 8:
            self.linear1 = Linear.DistributedXWT(
                in_dim//2, hidden_dim//4, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXWT(
                hidden_dim//2, out_dim//4, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank

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

    def __init__(self, in_dim, hidden_dim, out_dim, rank, device, n_parallel, model_group, bias=True, dropout=1e-3):
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
        rank : int
            Global rank.
        parallelism : int, optional
            The level of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        if n_parallel == 2:
            self.linear1 = Linear.DistributedXTW(
                in_dim, hidden_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXWT(
                hidden_dim//2, out_dim, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
        elif n_parallel == 4:
            self.linear1 = Linear.DistributedXTW(
                in_dim//2, hidden_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXW(
                hidden_dim//2, out_dim//2, rank, device, n_parallel, model_group, bias=bias)
        elif n_parallel == 8:
            self.linear1 = Linear.DistributedXTW(
                in_dim//4, hidden_dim//2, rank, device, n_parallel, model_group, bias=bias, init_method='linear')
            self.linear2 = Linear.DistributedXWT(
                hidden_dim//2, out_dim//4, rank, device, n_parallel, model_group, bias=bias)

        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank

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

    def __init__(self, in_dim, hidden_dim, out_dim, dropout=1e-3):
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
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, out_dim)
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

class DistributedMLP_Megatron(torch.nn.Module):
    """Distributed MLP across n processes."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: torch.distributed.ProcessGroup,
        bias: bool = True,
        dropout: float = 1e-3
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
        rank : int
            Global rank.
        parallelism : int, optional
            Degree of parallelism (default is 4).
        bias : bool, optional
            If True, adds a learnable bias to the output (default is True).
        dropout : int, optional
            dropout rate. Default value is 1e-3.
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        self.linear1 = torch.nn.Linear(in_dim, hidden_dim//n_parallel, bias=bias, device=None)
        self.linear2 = torch.nn.Linear(hidden_dim//n_parallel, out_dim, bias=bias, device=None)
        self.forward_all_reduce = Linear_Megatron.ForwardAllReduce(self.model_group)
        self.backward_all_reduce = Linear_Megatron.BackwardAllReduce(self.model_group)
        self.activation = GELU()
        self.drop = Dropout(p=dropout)
        self.drop2 = Dropout(p=dropout)
        self.rank = rank

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
        x = self.backward_all_reduce(x)
        x = self.linear1(x)
        
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = x.contiguous()
        # forward all reduce
        self.forward_all_reduce(x)
        
        return x