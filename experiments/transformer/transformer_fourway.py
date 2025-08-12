import Linear
import torch
from torch.nn import Softmax
from einops import rearrange
import utils


class DistributedAttention_fourway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, heads, rank, device, model_group=None):
    super().__init__()
    self.model_group = model_group
    self.heads = heads
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    self.n_parallel = 4
    
    self.qkv = Linear.DistributedXW(x_features, embed_dim*3,  rank, device, n_parallel=self.n_parallel)
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedMatMulXWTranspose(rank, device, self.n_parallel) # model group TODO 
    self.xw = Linear.DistributedMatMulXW(rank, device, self.n_parallel) # model group TODO 
    
    self.device = device
    self.rank = rank
    self.scale = embed_dim ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x, model_group):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(1, self.x_items, self.x_features) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
    qkv = self.qkv(x_patches_local, model_group) # shape: [n_batches, spatial_tokens, channel_tokens]
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    print(f'Here 1')
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    print(f'Here 2')
    q = self.scale * q
    
    attention = self.xwt(q, k, None, model_group)
    print(f'Here 3')
    attention = self.softmax(attention, model_group) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    print(f'Here 4')
    values = self.xw(attention, v, model_group)
    print(f'Here 5')
    return values
  

device, _, rank, _ = utils.init_CUDA()
model_group = utils.get_local_group_per_node(rank)

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

model = DistributedAttention_fourway(x_items, x_features, embed_dim, heads, rank, device, model_group=model_group).to(device)
target = (x[:,:,:2].clone()).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for i in range(10000):
  values = model(x, model_group)

  loss = torch.nn.functional.mse_loss(values, target)
  if rank == 0:
    print(loss)
  loss.backward()
  optimizer.step()
  
torch.distributed.destroy_process_group()