import torch
from einops import rearrange

import Attention
import Linear
import utils


class DistributedAttention_fourway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, rank, device, n_parallel, model_group, output_dim, heads):
    super().__init__()
    self.model_group = model_group
    self.heads = heads
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    self.n_parallel = 4
    
    self.qkv = Linear.DistributedXW(x_features, embed_dim*3,  rank, device, n_parallel=self.n_parallel, model_group=self.model_group)
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedMatMulXWTranspose(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedMatMulXW(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.concatenator = Linear.DistributedXW(embed_dim, output_dim, rank, device, self.n_parallel, self.model_group, n_channels=1, init_method='linear', bias=False)
    self.rank = rank
    self.device = device
    self.scale = embed_dim ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(1, self.x_items, self.x_features) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
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
    
    return values
  

device, _, rank, _ = utils.init_CUDA()
n_parallel=4
model_group = utils.get_model_group(rank, n_parallel)
softmax_group = utils.get_model_group_rows(rank, n_parallel, 2)

x_items, x_features, embed_dim, heads = 100, 8, 4, 2 # local values
torch.manual_seed(7)
x = torch.zeros((1, x_items, x_features)).to(device)
if rank == 0:
  x[:,:,0] = torch.arange(0, x_items) / x_items

  x[:,:,1] = x[:,:,0]**2
  x[:,:,2] = x[:,:,0]**3
  x[:,:,3] = x[:,:,0]**2+0.01
else:
  x[:,:,0] = torch.arange(0, x_items) / x_items
  x[:,:,1] = x[:,:,0]+0.1
  x[:,:,2] = x[:,:,0]-0.1
  x[:,:,3] = x[:,:,0]*1.1


batch = 1
x_items, x_features, embed_dim, heads = 4, 8, 16, 4 # x_features is global; embed_dim is global
output_dim = 6
batch = 1

model = Attention.DistributedAttention(
  x_features, 
  embed_dim, 
  output_dim, # global
  heads,
  rank, 
  device,
  n_parallel=n_parallel, 
  model_group=model_group,
  softmax_group=softmax_group
).to(torch.float32).to(device)

  
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

X = torch.rand(1, x_features//2, x_features//2).to(device)

for i in range(1000):
  
  values = model(X)
  loss = torch.nn.MSELoss()(values, values*0.1)
  
  if rank == 0:
    print(loss)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  
torch.distributed.destroy_process_group()