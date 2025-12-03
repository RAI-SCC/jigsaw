import torch

from . import MLP, Linear
from . import Normalization as Norm


class MixerBlock(torch.nn.Module):
    """Mixing block that performs token and channel mixing using four-way distributed MLPs and layer normalization."""

    def __init__(
        self,
        tokens_mlp_dim,
        channels_mlp_dim,
        spatial_hidden_dim_fraction,
        channels_hidden_dim_fraction,
        rank,
        device,
        n_parallel,
        model_group,
    ):
        """
        Parameters
        ----------
        tokens_mlp_dim : int
            Global dimension of the tokens MLP.
        channels_mlp_dim : int
            Global dimension of the channels MLP.
        spatial_hidden_dim_fraction : float
            Fraction of the hidden dimension for spatial mixing.
        channels_hidden_dim_fraction : float
            Fraction of the hidden dimension for channel mixing.
        device : torch.device
            Device to run the computations on.
        rank : int
            Global rank.
        """
        super(MixerBlock, self).__init__()
        self.token_mixing = MLP.DistributedXTWMLP(
            tokens_mlp_dim,
            int(tokens_mlp_dim * spatial_hidden_dim_fraction),
            tokens_mlp_dim,
            rank,
            device,
            n_parallel=n_parallel,
            model_group=model_group,
            bias=True
        )
        self.channel_mixing = MLP.DistributedMLP(
            channels_mlp_dim,
            int(channels_mlp_dim * channels_hidden_dim_fraction),
            channels_mlp_dim,
            rank,
            device,
            n_parallel=n_parallel,
            model_group=model_group,
            bias=True
        )
        self.transpose = Linear.DistributedXTranpose(
            rank, device, n_parallel, model_group
        )
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group

        if n_parallel == 2:
            self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim // 2)
            self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim // 2)
        elif n_parallel == 4 or n_parallel == 8:
            self.layer_norm1 = Norm.DistributedLayerNorm(
                channels_mlp_dim // 2, self.rank, self.device, self.model_group
            )
            self.layer_norm2 = Norm.DistributedLayerNorm(
                channels_mlp_dim // 2, self.rank, self.device, self.model_group
            )

    def forward(self, x):
        """
        Forward pass of MixerBlock distributed across two processes.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, n_patches, embed_dimension)
            = shape (B, x_in, in_dim).
        model_group : torch.distributed.ProcessGroup, optional
            Process group for an individual model-parallel instance.

        Returns
        -------
        torch.Tensor
            Output tensor after applying layer normalization, token mixing,
            transposition, and channel mixing.
        """
        x_res = x
        y = self.layer_norm1(x)
        y = self.token_mixing(y)
        y = self.transpose(y)
        x = x_res + y
        y = self.layer_norm2(x)
        x = x + self.channel_mixing(y)
        
        return x


class MixerBlockSequential(torch.nn.Module):
    """
    A sequential Mixer block for token and channel mixing.

    Parameters.
    ----------
    tokens_mlp_dim : int
        Dimension of the token MLP.
    channels_mlp_dim : int
        Dimension of the channel MLP.
    spatial_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the token MLP.
    channels_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the channel MLP.
    dropout : float, optional
        Dropout probability, by default 1e-3.
    """

    def __init__(
        self,
        tokens_mlp_dim,
        channels_mlp_dim,
        spatial_hidden_dim_fraction,
        channels_hidden_dim_fraction,
        dropout=1e-3,
    ):
        super(MixerBlockSequential, self).__init__()

        self.token_mixing = MLP.SequentialMLP(
            tokens_mlp_dim,
            int(tokens_mlp_dim * spatial_hidden_dim_fraction),
            tokens_mlp_dim,
            dropout=dropout,
        )
        self.channel_mixing = MLP.SequentialMLP(
            channels_mlp_dim,
            int(channels_mlp_dim * channels_hidden_dim_fraction),
            channels_mlp_dim,
            dropout=dropout,
        )
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)

    def forward(self, x):
        """
        Forward pass of MixerBlock.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, n_patches, embed_dimension)
            = shape (B, x_in, in_dim).

        Returns
        -------
        x : torch.Tensor
            Output tensor after applying layer normalization, token mixing, and channel mixing.
        """
        x_res = x.clone()
        y = self.layer_norm1(x)
        y = y.mT
        y = self.token_mixing(y)
        y = y.mT
        x = x_res + y
        y = self.layer_norm2(x)
        x = x + self.channel_mixing(y)

        return x


class MixerBlockMegatron(torch.nn.Module):
    """
    A sequential Mixer block for token and channel mixing.

    Parameters.
    ----------
    tokens_mlp_dim : int
        Dimension of the token MLP.
    channels_mlp_dim : int
        Dimension of the channel MLP.
    spatial_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the token MLP.
    channels_hidden_dim_fraction : float
        Fraction to determine the hidden dimension size for the channel MLP.
    dropout : float, optional
        Dropout probability, by default 1e-3.
    """

    def __init__(
        self,
        tokens_mlp_dim,
        channels_mlp_dim,
        spatial_hidden_dim_fraction,
        channels_hidden_dim_fraction,
        rank,
        device,
        n_parallel,
        model_group,
        dropout=1e-3,
        bias=True
    ):
        super(MixerBlockMegatron, self).__init__()
        self.token_mixing = MLP.DistributedMLP_Megatron(
            tokens_mlp_dim,
            int(tokens_mlp_dim * spatial_hidden_dim_fraction),
            tokens_mlp_dim,
            rank,
            device,
            n_parallel,
            model_group,
            bias,
            dropout,
        )
        self.channel_mixing = MLP.DistributedMLP_Megatron(
            channels_mlp_dim,
            int(channels_mlp_dim * channels_hidden_dim_fraction),
            channels_mlp_dim,
            rank,
            device,
            n_parallel,
            model_group,
            bias,
            dropout,
        )
        self.layer_norm1 = torch.nn.LayerNorm(channels_mlp_dim)
        self.layer_norm2 = torch.nn.LayerNorm(channels_mlp_dim)

    def forward(self, x):
        """
        Forward pass of MixerBlock.

        Parameters
        ----------
        x : torch.Tensor
            Local input tensor of shape (B, n_patches, embed_dimension)
            = shape (B, x_in, in_dim).

        Returns
        -------
        x : torch.Tensor
            Output tensor after applying layer normalization, token mixing, and channel mixing.
        """
        x_res = x.clone()
        y = self.layer_norm1(x)
        y = y.mT
        y = self.token_mixing(y)
        y = y.mT
        x = x_res + y
        y = self.layer_norm2(x)
        x = x + self.channel_mixing(y)

        return x
