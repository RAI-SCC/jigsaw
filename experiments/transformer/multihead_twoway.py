import Linear
import torch
from torch.nn import Softmax
import utils
import distributed_matmul as dist_mm
from einops import rearrange

class DistributedAttention_twoway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, device, rank, output_dim, heads=1, model_group=None):
    super().__init__()
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.qkv = Linear.DistributedXW_twoway(x_features, embed_dim//2*3, device, rank, n_channels=1, init_method='linear', bias=False)
    
    #with torch.no_grad():
    #  self.qkv.bias.copy_(torch.zeros(2).to(device))
      
#
    #if rank == 0:
    #  with torch.no_grad():
    #    self.q.weights.copy_(
    #    torch.tensor([
    #      [0.7204, 0.0731],
    #      [0.8829, 0.4132],
    #      [0.5209, 0.5932],
    #      [0.7653, 0.1132]])
    #    )
    #    self.k.weights.copy_(
    #      torch.tensor([
    #        [0.6267, 0.5691],
    #        [0.3887, 0.2214],
    #        [0.7405, 0.2529],
    #        [0.9575, 0.5575]]).to(device)
    #    )
    #    self.v.weights.copy_(
    #      torch.tensor([
    #        [0.7369, 0.0331],
    #        [0.9936, 0.4703],
    #        [0.2674, 0.4990],
    #        [0.4414, 0.5550]]
    #      ).to(device)
    #    )
    #elif rank == 1:
    #  with torch.no_grad():
    #    self.q.weights.copy_(
    #    torch.tensor([
    #      [0.9699, 0.1078],
    #      [0.7572, 0.6948],
    #      [0.8797, 0.6286],
    #      [0.8559, 0.6721]]).to(device)
    #    )
    #    self.k.weights.copy_(
    #      torch.tensor([
    #        [0.7437, 0.9592],
    #        [0.3742, 0.1953],
    #        [0.2332, 0.9314],
    #        [0.4134, 0.4355]
    #      ]).to(device)
    #    )
    #    self.v.weights.copy_(
    #      torch.tensor([
    #        [0.0914, 0.8994],
    #        [0.1049, 0.5137],
    #        [0.7447, 0.7213],
    #        [0.6361, 0.1081]
    #      ]).to(device)
    #    )
      
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedXWT_attention_twoway(device, rank, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedXW_attention_twoway(device, rank, model_group=self.model_group) # model group TODO 
    
    self.concatenator = Linear.DistributedXW_twoway(embed_dim, output_dim//2, device, rank, n_channels=1, init_method='linear', bias=False)
    self.mlp = Linear.DistributedMLP(output_dim, output_dim, output_dim, device, rank, parallelism=2)
    self.mlp2 = Linear.DistributedMLP(output_dim, output_dim, output_dim, device, rank, parallelism=2)
    self.device = device
    self.rank = rank
    self.heads = heads
    self.scale = (embed_dim // self.heads) ** (-0.5) # should be (embed_dim/heads)
    

  def forward(self, x, model_group=None):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x

    qkv = self.qkv(x_patches_local, model_group) # shape: [n_batches, spatial_tokens, channel_tokens]
    #print(f'qkv {qkv}')
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    #q, k, v = torch.tensor_split(q, self.heads, dim=2), torch.tensor_split(k, self.heads, dim=2), torch.tensor_split(v, self.heads, dim=2)
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads)
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads)
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads)
    q = self.scale * q
    
    #print(f'k {k}')
    attention = self.xwt(q, k)
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    
    values = self.xw(attention, v)    
    values = rearrange(values, 'n h s d -> n s (h d)')
    values = self.concatenator(values, model_group=None)
    values = self.mlp(values, model_group=None)
    values = self.mlp2(values, model_group=None)
    # Output: [n_batch, seq_length, output_dim]
    return values



device, _, rank, _ = utils.init_CUDA()

batch = 1000
x_items, x_features, embed_dim, heads = 4, 4, 64*4, 8 # x_features is global; embed_dim is global
torch.manual_seed(7)
W = torch.randn(x_features, x_features//2).to(device)
#b = torch.randn(x_features//2).to(device)

def sample_batch():
    x = torch.randn(batch, x_items, x_features//2).to(device)
    y = dist_mm.XW_twoway(x, W, device, rank)  # (batch, items, features)
    return x, y


#if rank == 0:
#  x = torch.tensor([
#    [0.5349, 0.1988],
#    [0.2328, 0.4251],
#    [0.3653, 0.8513],
#    [0.2868, 0.2063]]).to(device)
#
#elif rank == 1:
#  x = torch.tensor([
#    [0.6592, 0.6569],
#    [0.2071, 0.6297],
#    [0.8549, 0.5509],
#    [0.4451, 0.3593]
#  ]).to(device)

model = DistributedAttention_twoway(
  x_items, 
  x_features, 
  embed_dim, 
  device, 
  rank, 
  output_dim=x_features, # global
  heads=heads, 
  model_group=None
).to(torch.float32).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
x, target = sample_batch()  
  
for i in range(10000):
  
  values = model(x)

  loss = torch.nn.functional.l1_loss(values, target)
  if rank == 0:
    print(loss)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
# Resulting global product should have shape 4, 12


#result = model(x)

torch.distributed.destroy_process_group()