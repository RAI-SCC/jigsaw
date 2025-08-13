import numpy as np
import torch
from einops import rearrange
from torch.nn import GELU, Dropout
import Linear
import DistributedNormFunctions as Norm

class DistributedAttention(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, output_dim, heads, rank, device, n_parallel, model_group):
    super().__init__()
    self.rank = rank
    self.device = device
    self.n_parallel = n_parallel
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    if n_parallel == 2:
        self.qkv = Linear.DistributedXW(x_features, embed_dim//2*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    elif n_parallel == 4:
        self.qkv = Linear.DistributedXW(x_features, embed_dim*3, self.rank, self.device, self.n_parallel, self.model_group,  n_channels=1, init_method='linear', bias=False)
    self.softmax = Norm.DistributedSoftMax(dim=-1, group=self.model_group) # TODO: define softmax groups
    self.xwt = Linear.DistributedMatMulXWTranspose(self.rank, self.device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedMatMulXW(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    
    if n_parallel == 2:
        self.concatenator = Linear.DistributedXW(embed_dim, output_dim//2, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    else:
        self.concatenator = Linear.DistributedXW(embed_dim, output_dim, self.rank, self.device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    
    self.heads = heads
    self.scale = (embed_dim // self.heads) ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x):
    """
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
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    values = self.xw(attention, v, None)    
    values = rearrange(values, 'n h s d -> n s (h d)').contiguous()
    
    values = self.concatenator(values)
    #values = self.mlp(values, model_group=None)
    # Output: [n_batch, seq_length, output_dim]
    return values