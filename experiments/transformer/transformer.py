import Linear
import torch
from torch.nn import Softmax
import utils

class DistributedAttention_twoway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, device, rank, model_group=None):
    super().__init__()
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.q = Linear.DistributedXWT_twoway(x_features//2, embed_dim,  device, rank, n_channels=1, init_method='linear')
    self.k = Linear.DistributedXWT_twoway(x_features//2, embed_dim,  device, rank, n_channels=1, init_method='linear')
    self.v = Linear.DistributedXW_twoway(x_features, embed_dim,  device, rank, n_channels=1, init_method='linear')
    
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedXWT_attention_twoway(device, rank, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedXW_attention_twoway(device, rank, model_group=self.model_group) # model group TODO 
    
    self.mlp = Linear.DistributedMLP(embed_dim*2, embed_dim*6, 2, device, rank, parallelism=2)
    self.device = device
    self.rank = rank
    self.scale = embed_dim ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x, model_group=None):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(1, self.x_items, self.x_features//2) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
    q = self.q(x_patches_local, model_group) # shape: [n_batches, spatial_tokens, channel_tokens]
    k = self.k(x_patches_local, model_group)
    v = self.v(x_patches_local, model_group)
    
    q = self.scale * q
    
    attention = self.xwt(q, k)
    

    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    
    values = self.xw(attention, v)    
    values = self.mlp(values, model_group=None)
    
    # values shape [x_length, ]

    return values


device, _, rank, _ = utils.init_CUDA()

x_items, x_features, embed_dim = 10, 8, 4 # global values
torch.manual_seed(7)
x = torch.zeros((1, x_items, x_features//2)).to(device)
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
print(x[:5])

attention = DistributedAttention_twoway(x_items, x_features, embed_dim, device, rank, model_group=None).to(device)
target = (x[:,:,0]).to(device)

optimizer = torch.optim.Adam(attention.parameters(), lr=0.00001)
for i in range(10000):
  values = attention(x)

  loss = torch.nn.functional.mse_loss(values, target)
  if rank == 0:
    print(loss)
  loss.backward()
  optimizer.step()
  


torch.distributed.destroy_process_group()