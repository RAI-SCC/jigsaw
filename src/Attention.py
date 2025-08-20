import torch
from einops import rearrange

import Linear
import Normalization as Norm


class DistributedAttention(torch.nn.Module):
  def __init__(
        self, 
        x_features: int, 
        embed_dim: int, 
        output_dim: int, 
        heads: int, 
        rank: int, 
        device: torch.device, 
        n_parallel: int, 
        model_group: torch.distributed.ProcessGroup, 
        softmax_group: torch.distributed.ProcessGroup
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
    
    if n_parallel == 2:
        self.qkv = Linear.DistributedXWT(x_features//2, embed_dim*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 4:
        self.qkv = Linear.DistributedXWT(x_features//2, embed_dim//2*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 8:
        self.qkv = Linear.DistributedXWT(x_features//4, embed_dim//2*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    
    self.softmax = Norm.DistributedSoftMax(dim=-1, group=self.softmax_group) # TODO: define softmax groups
    self.xwt = Linear.DistributedMatMulXWTranspose(self.rank, self.device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedMatMulXW(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    
    if n_parallel == 2:
        self.concatenator = Linear.DistributedXWT(embed_dim//2, output_dim, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 4:
        self.concatenator = Linear.DistributedXWT(embed_dim//2, output_dim//2, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 8:
        self.concatenator = Linear.DistributedXWT(embed_dim//2, output_dim//4, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    
    self.heads = heads
    self.scale = (embed_dim * 2 // self.heads) ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass for distributed multi-head attention.
    
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x

    qkv = self.qkv(x_patches_local) # shape: [n_batches, spatial_tokens, channel_tokens]
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    
    q = self.scale * q
    attention = self.xwt(q, k, None)
    attention = self.softmax(attention)
    values = self.xw(attention, v, None)    
    values = rearrange(values, 'n h s d -> n s (h d)').contiguous()
    
    values = self.concatenator(values)

    # Output: [n_batch, seq_length, output_dim]
    return values