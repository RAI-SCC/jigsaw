import Linear
import torch
from torch.nn import Softmax
from einops import rearrange
import utils


class DistributedAttention_fourway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, heads, device, rank, model_group=None):
    super().__init__()
    self.model_group = model_group
    self.heads = heads
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.qkv = Linear.DistributedXW_fourway(x_features, embed_dim*3,  device, rank)
    
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedXWT_attention_fourway(device, rank, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedXW_attention_fourway(device, rank, model_group=self.model_group) # model group TODO 
    
    self.device = device
    self.rank = rank
    self.scale = embed_dim ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x, model_group=None):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(1, self.x_items, self.x_features) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
    qkv = self.qkv(x_patches_local, model_group) # shape: [n_batches, spatial_tokens, channel_tokens]
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads).contiguous()
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads).contiguous()

    q = self.scale * q
    
    attention = self.xwt(q, k)
    
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    
    values = self.xw(attention, v)    

    return values
  

device, _, rank, _ = utils.init_CUDA()

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

model = DistributedAttention_fourway(x_items, x_features, embed_dim, heads, device, rank, model_group=None).to(device)
target = (x[:,:,:2].clone()).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for i in range(10000):
  values = model(x)

  loss = torch.nn.functional.mse_loss(values, target)
  if rank == 0:
    print(loss)
  loss.backward()
  optimizer.step()
  
torch.distributed.destroy_process_group()