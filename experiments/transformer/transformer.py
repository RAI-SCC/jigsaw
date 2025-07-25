import Linear
import torch
from torch.nn import Softmax
import utils
import distributed_matmul as dist_mm

class DistributedAttention_twoway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, device, rank, model_group=None):
    super().__init__()
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.q = Linear.DistributedXW_twoway(x_features, embed_dim//2,  device, rank, n_channels=1, init_method='linear')
    self.k = Linear.DistributedXW_twoway(x_features, embed_dim//2,  device, rank, n_channels=1, init_method='linear')
    self.v = Linear.DistributedXW_twoway(x_features, embed_dim//2,  device, rank, n_channels=1, init_method='linear')
    
    with torch.no_grad():
      self.q.bias.copy_(torch.zeros(2).to(device))
      self.k.bias.copy_(torch.zeros(2).to(device))
      self.v.bias.copy_(torch.zeros(2).to(device))

    if rank == 0:
      with torch.no_grad():
        self.q.weights.copy_(
        torch.tensor([
          [0.7204, 0.0731],
          [0.8829, 0.4132],
          [0.5209, 0.5932],
          [0.7653, 0.1132]])
        )
        self.k.weights.copy_(
          torch.tensor([
            [0.6267, 0.5691],
            [0.3887, 0.2214],
            [0.7405, 0.2529],
            [0.9575, 0.5575]]).to(device)
        )
        self.v.weights.copy_(
          torch.tensor([
            [0.7369, 0.0331],
            [0.9936, 0.4703],
            [0.2674, 0.4990],
            [0.4414, 0.5550]]
          ).to(device)
        )
    elif rank == 1:
      with torch.no_grad():
        self.q.weights.copy_(
        torch.tensor([
          [0.9699, 0.1078],
          [0.7572, 0.6948],
          [0.8797, 0.6286],
          [0.8559, 0.6721]]).to(device)
        )
        self.k.weights.copy_(
          torch.tensor([
            [0.7437, 0.9592],
            [0.3742, 0.1953],
            [0.2332, 0.9314],
            [0.4134, 0.4355]
          ]).to(device)
        )
        self.v.weights.copy_(
          torch.tensor([
            [0.0914, 0.8994],
            [0.1049, 0.5137],
            [0.7447, 0.7213],
            [0.6361, 0.1081]
          ]).to(device)
        )
      
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
    x_patches_local = x.unsqueeze(0)

    q = self.q(x_patches_local, model_group) # shape: [n_batches, spatial_tokens, channel_tokens]
    k = self.k(x_patches_local, model_group)
    v = self.v(x_patches_local, model_group)
    
    q = self.scale * q
    
    attention = self.xwt(q, k)
    
    
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    print(f'Rank {rank}: softmax {attention}')
    values = self.xw(attention, v)    
    print(values)
    #values = self.mlp(values, model_group=None)
    
    # values shape [x_length, ]

    return values


device, _, rank, _ = utils.init_CUDA()

x_items, x_features, embed_dim = 4, 4, 4 # x_features is global; embed_dim is global
torch.manual_seed(7)
x = torch.zeros((1, x_items, x_features)).to(device)
if rank == 0:
  x = torch.tensor([
    [0.5349, 0.1988],
    [0.2328, 0.4251],
    [0.3653, 0.8513],
    [0.2868, 0.2063]]).to(device)

elif rank == 1:
  x = torch.tensor([
    [0.6592, 0.6569],
    [0.2071, 0.6297],
    [0.8549, 0.5509],
    [0.4451, 0.3593]
  ]).to(device)
model = DistributedAttention_twoway(x_items, x_features, embed_dim, device, rank, model_group=None).to(device)


#result = model(x)

#optimizer = torch.optim.Adam(attention.parameters(), lr=0.00001)
#for i in range(1):
#  values = attention(x)
#
#  loss = torch.nn.functional.mse_loss(values, target)
#  if rank == 0:
#    print(loss)
#  loss.backward()
#  optimizer.step()
  
# Resulting global product should have shape 4, 12
x = torch.rand((4,2)).to(device) 
w = torch.rand((3, 4,6)).to(device) 
result = dist_mm.XW_twoway(x, w, device, rank, model_group=None).requires_grad_(True)
target = torch.rand((3, 4, 6)).to(device)
print(result)
loss = torch.nn.functional.mse_loss(result, target)

torch.distributed.destroy_process_group()