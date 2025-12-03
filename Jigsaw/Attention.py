import torch
import torch.distributed as dist
from einops import rearrange

import Jigsaw.Normalization as Norm
from Jigsaw import MLP, Linear, Linear_Megatron


class DistributedTransformer(torch.nn.Module):
    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        softmax_group: dist.ProcessGroup,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.softmax_group = softmax_group

        self.embed_dim = embed_dim
        self.x_features = x_features

        match n_parallel:
            case 1:
                self.qkv = torch.nn.Linear(x_features, embed_dim * 3, bias=False)
            case 2:
                self.qkv = Linear.DistributedXWT(
                    x_features // 2,
                    embed_dim * 3,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )
            case 4:
                self.qkv = Linear.DistributedXWT(
                    x_features // 2,
                    embed_dim // 2 * 3,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )
            case 8:
                self.qkv = Linear.DistributedXWT(
                    x_features // 4,
                    embed_dim // 2 * 3,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )

        if self.n_parallel > 1:
            self.softmax = Norm.DistributedSoftMax(
                dim=-1, group=self.softmax_group
            )  # TODO: define softmax groups
            self.xwt = Linear.DistributedMatMulXWTranspose(
                self.rank, self.device, self.n_parallel, model_group=self.model_group
            )  # model group TODO
            self.xw = Linear.DistributedMatMulXW(
                rank, device, self.n_parallel, model_group=self.model_group
            )  # model group TODO
        else:
            self.softmax = torch.nn.Softmax(dim=-1)
            self.xwt = torch.nn.Identity() # Placeholder for non-parallel case
            self.xw = torch.nn.Identity()

        match self.n_parallel:
            case 1:
                self.concatenator = torch.nn.Linear(embed_dim, output_dim, bias=False)
            case 2:
                self.concatenator = Linear.DistributedXWT(
                    embed_dim // 2,
                    output_dim,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )
            case 4:
                self.concatenator = Linear.DistributedXWT(
                    embed_dim // 2,
                    output_dim // 2,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )
            case 8:
                self.concatenator = Linear.DistributedXWT(
                    embed_dim // 2,
                    output_dim // 4,
                    self.rank,
                    self.device,
                    self.n_parallel,
                    self.model_group,
                    n_channels=1,
                    init_method="linear",
                    bias=False,
                )

        self.heads = heads
        # should be (embed_dim/heads)
        self.scale = (embed_dim * 2 // self.heads) ** (-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # Load constant masks from the disc

        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x

        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.qkv(x_patches_local)
        q, k, v = torch.tensor_split(qkv, 3, dim=2)

        q = rearrange(q, "n i (h d) -> n h i d", h=self.heads).contiguous()
        k = rearrange(k, "n i (h d) -> n h i d", h=self.heads).contiguous()
        v = rearrange(v, "n i (h d) -> n h i d", h=self.heads).contiguous()

        q = self.scale * q
        if self.n_parallel > 1:
            attention = self.xwt(q, k, None)
            attention = self.softmax(attention)
            values = self.xw(attention, v, None)
        else:
            attention = q @ k.mT
            attention = self.softmax(attention)
            values = attention @ v
            
        values = rearrange(values, "n h s d -> n s (h d)").contiguous()

        values = self.concatenator(values)

        # Output: [n_batch, seq_length, output_dim]
        return values

class TransformerLayer(torch.nn.Module):
    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        softmax_group: dist.ProcessGroup,
        column_subgroup: dist.ProcessGroup
    ):
        super().__init__()
        self.attention_block = DistributedTransformer(
            x_features,
            embed_dim,
            output_dim,
            heads,
            rank,
            device,
            n_parallel,
            model_group,
            softmax_group
        )
        self.layer_norm = Norm.DistributedLayerNorm(
            output_dim // 2,
            rank,
            device,
            column_subgroup
        )
        self.MLP = MLP.DistributedMLP(
            output_dim,
            output_dim // 2,
            output_dim,
            rank,
            device,
            n_parallel,
            model_group
        )
        
    def forward(self, x: torch.tensor):
        x = self.attention_block(x)
        x = self.layer_norm(x)
        x = self.MLP(x)
        return x
    
class MegatronTransformer(torch.nn.Module):
    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        softmax_group: dist.ProcessGroup,
    ):
        """

        Torch Module to perform distributed multi-headed attention.

        x_features: int
            global number of x features
        embed_dim: int
            global embedding dimension
        output_dim: int
            global
        """
        super().__init__()
        self.rank = rank
        self.device = device
        self.n_parallel = n_parallel
        self.model_group = model_group
        self.softmax_group = softmax_group
        self.bias = False

        self.embed_dim = embed_dim
        self.x_features = x_features

        self.qkv = torch.nn.Linear(x_features, embed_dim * 3, bias=self.bias, device=None)
        self.concatenator = torch.nn.Linear(embed_dim, output_dim, bias=self.bias, device=None)
        self.forward_all_reduce = Linear_Megatron.ForwardAllReduce(self.model_group)
        self.backward_all_reduce = Linear_Megatron.BackwardAllReduce(self.model_group)

        self.softmax = torch.nn.Softmax(dim=-1)

        self.heads = heads // n_parallel
        # should be (embed_dim/heads)
        self.scale = (embed_dim * 2 // self.heads) ** (-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for distributed multi-head attention.

        x: (n_batch, vars, n_lat, n_lon)
        """
        # Load constant masks from the disc

        # x_local: [n_batch, n_lat, n_lon, variables]
        x_patches_local = x
        
        # shape: [n_batches, spatial_tokens, channel_tokens]
        qkv = self.backward_all_reduce(x_patches_local)
        qkv = self.qkv(qkv)
        
        q, k, v = torch.tensor_split(qkv, 3, dim=2)

        q = rearrange(q, "n i (h d) -> n h i d", h=self.heads).contiguous()
        k = rearrange(k, "n i (h d) -> n h i d", h=self.heads).contiguous()
        v = rearrange(v, "n i (h d) -> n h i d", h=self.heads).contiguous()

        q = self.scale * q
        # print(f'q shape {q.shape} kmt shape {(k.mT).shape}')
        attention = q @ k.mT
        attention = self.softmax(attention)
        # print(f'attention after softmax {attention.shape}, v shape {v.shape}')
        values = attention @ v

        values = rearrange(values, "n h s d -> n s (h d)").contiguous()

        values = self.concatenator(values)
        values = values.contiguous()
        # forward all reduce
        self.forward_all_reduce(values)
        # Output: [n_batch, seq_length, output_dim]
        return values
    
class MegatronTransformerLayer(torch.nn.Module):
    def __init__(
        self,
        x_features: int,
        embed_dim: int,
        output_dim: int,
        heads: int,
        rank: int,
        device: torch.device,
        n_parallel: int,
        model_group: dist.ProcessGroup,
        softmax_group: dist.ProcessGroup,
        column_subgroup: dist.ProcessGroup
    ):
        super().__init__()
        self.attention_block = MegatronTransformer(
            x_features,
            embed_dim,
            output_dim,
            heads,
            rank,
            device,
            n_parallel,
            model_group,
            softmax_group
        )
        self.layer_norm = torch.nn.LayerNorm(
            output_dim
        )
        self.MLP = MLP.DistributedMLP_Megatron(
            output_dim,
            output_dim // 2,
            output_dim,
            rank,
            device,
            n_parallel,
            model_group
        )
        
    def forward(self, x: torch.tensor):
        x = self.attention_block(x)
        x = self.layer_norm(x)
        x = self.MLP(x)
        return x