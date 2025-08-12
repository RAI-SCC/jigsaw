import Linear
import torch
from torch.nn import Softmax
import utils
import distributed_matmul as dist_mm
from einops import rearrange
from sklearn.datasets import make_blobs
from sklearn.preprocessing import normalize
from typing import Tuple
import numpy as np

class DistributedAttention_twoway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, device, rank, output_dim, heads=1, model_group=None):
    super().__init__()
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.qkv = Linear.DistributedXW(2, x_features, embed_dim//2*3, device, rank, n_channels=1, init_method='linear', bias=False)
    self.concatenator = Linear.DistributedXW(2, embed_dim, output_dim//2, device, rank, n_channels=1, init_method='linear', bias=False)
    #with torch.no_grad():
    #  self.qkv.bias.copy_(torch.zeros(6).to(device))
    #  self.concatenator.bias.copy_(torch.zeros(3).to(device))
      
#
    if rank == 0:
      with torch.no_grad():
        self.qkv.weights.copy_(
        torch.tensor([[ 0.0349, -0.2672,  0.2204,  0.3829,  0.1267, -0.1113],
          [-0.3012, -0.0749, -0.4269, -0.0868,  0.0691, -0.2786],
          [ 0.1592, -0.2929,  0.4699,  0.2572,  0.2437, -0.1258],
          [ 0.1569,  0.1297, -0.3922,  0.1948,  0.4592, -0.3047]]).to(device)
        )
        self.concatenator.weights.copy_(
          torch.tensor([[ 0.2369,  0.4936, -0.2326],
          [-0.4669, -0.0297, -0.0010],
          [-0.4086, -0.3951,  0.2447],
          [ 0.3994,  0.0137,  0.2213]]).to(device)
        )
        
        
    elif rank == 1:
      with torch.no_grad():
        self.qkv.weights.copy_(
        torch.tensor([[-0.1347, -0.2132,  0.0209,  0.2653,  0.2405,  0.4575],
          [ 0.3513, -0.2937,  0.0932, -0.3868, -0.2471,  0.0575],
          [ 0.3549, -0.0549,  0.3797,  0.3559, -0.2668, -0.0866],
          [ 0.0509, -0.1407,  0.1286,  0.1721,  0.4314, -0.0645]]).to(device)
        )
        
        self.concatenator.weights.copy_(
          torch.tensor([[-0.0586, -0.1695,  0.1679],
          [ 0.0550,  0.0196,  0.2878],
          [ 0.1361, -0.2853,  0.0070],
          [-0.3919, -0.2184, -0.1945]]).to(device)
        )
    self.concatenator = Linear.DistributedXW(2, embed_dim, output_dim//2, device, rank, n_channels=1, init_method='linear', bias=False)
    #with torch.no_grad():
    #  self.qkv.bias.copy_(torch.zeros(6).to(device))
    #  self.concatenator.bias.copy_(torch.zeros(3).to(device))
      
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
    if rank == 0:
      with torch.no_grad():
        self.qkv.weights.copy_(
        torch.tensor([[ 0.0349, -0.2672,  0.2204,  0.3829,  0.1267, -0.1113],
          [-0.3012, -0.0749, -0.4269, -0.0868,  0.0691, -0.2786],
          [ 0.1592, -0.2929,  0.4699,  0.2572,  0.2437, -0.1258],
          [ 0.1569,  0.1297, -0.3922,  0.1948,  0.4592, -0.3047]]).to(device)
        )
        self.concatenator.weights.copy_(
          torch.tensor([[ 0.2369,  0.4936, -0.2326],
          [-0.4669, -0.0297, -0.0010],
          [-0.4086, -0.3951,  0.2447],
          [ 0.3994,  0.0137,  0.2213]]).to(device)
        )
        
        
    elif rank == 1:
      with torch.no_grad():
        self.qkv.weights.copy_(
        torch.tensor([[-0.1347, -0.2132,  0.0209,  0.2653,  0.2405,  0.4575],
          [ 0.3513, -0.2937,  0.0932, -0.3868, -0.2471,  0.0575],
          [ 0.3549, -0.0549,  0.3797,  0.3559, -0.2668, -0.0866],
          [ 0.0509, -0.1407,  0.1286,  0.1721,  0.4314, -0.0645]]).to(device)
        )
        
        self.concatenator.weights.copy_(
          torch.tensor([[-0.0586, -0.1695,  0.1679],
          [ 0.0550,  0.0196,  0.2878],
          [ 0.1361, -0.2853,  0.0070],
          [-0.3919, -0.2184, -0.1945]]).to(device)
        )

    self.n_parallel = 2
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedMatMulXWTranspose(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    self.xw = Linear.DistributedMatMulXW(rank, device, self.n_parallel, model_group=self.model_group) # model group TODO 
    
    
    #self.mlp = Linear.DistributedMLP(output_dim, output_dim, output_dim, device, rank, parallelism=2)
    self.concatenator = Linear.DistributedXW(embed_dim, output_dim//2, rank, device, self.n_parallel, n_channels=1, init_method='linear', bias=False)
    self.mlp = Linear.DistributedMLP(output_dim, output_dim, output_dim, rank, device, self.n_parallel)
    self.mlp2 = Linear.DistributedMLP(output_dim, output_dim, output_dim, rank, device, self.n_parallel)
    
    #self.mlp = Linear.DistributedMLP(output_dim, output_dim, output_dim, device, rank, parallelism=2)
    self.rank = rank
    self.device = device
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
    
    q, k, v = torch.tensor_split(qkv, 3, dim=2)
    
    q = rearrange(q, 'n i (h d) -> n h i d', h=self.heads)
    k = rearrange(k, 'n i (h d) -> n h i d', h=self.heads)
    v = rearrange(v, 'n i (h d) -> n h i d', h=self.heads)
    
    q = self.scale * q
    
    attention = self.xwt(q, k)
    
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    

    values = self.xw(attention, v)    
    
    values = rearrange(values, 'n h s d -> n s (h d)').contiguous()
    
    values = self.concatenator(values, model_group=None)
    print(values)
    #values = self.mlp(values, model_group=None)
    # Output: [n_batch, seq_length, output_dim]
    return values



device, _, rank, _ = utils.init_CUDA()
batch = 1
x_items, x_features, embed_dim, heads = 4, 4, 4, 2 # x_features is global; embed_dim is global
output_dim = 6
batch = 1
torch.manual_seed(7)
W = torch.randn(x_features, x_features//2).to(device)
#b = torch.randn(x_features//2).to(device)

def sample_batch():
    x = torch.randn(batch, x_items, x_features//2).to(device)
    y = dist_mm.XW_twoway(x, W, device, rank)  # (batch, items, features)
    
    return x, y

def generate_spectral_clustering_data(
    n_samples: int = 12, 
    n_clusters: int = 4, 
    cluster_std: float = 0.60,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate dummy data for spectral clustering.

    Args:
        n_samples: Total number of samples.
        n_clusters: Number of clusters.
        cluster_std: Standard deviation of clusters.
        random_state: Seed for reproducibility.

    Returns:
        X: (n_samples, 2) array of 2D data points.
        y: (n_samples,) array of ground truth cluster labels.
    """
    torch.manual_seed(random_state)
    X, y = make_blobs(n_samples=n_samples, centers=n_clusters, 
                      cluster_std=cluster_std, random_state=random_state)
    
    X = normalize(X)  # Normalize for similarity-based methods
    
    X = torch.tensor(X).to(device)
    y = np.eye(n_clusters)[y]
    y = torch.tensor(y).to(device)
    
    if rank == 0:
       X = X[:, 0]
       y = y[:, :2]
       
    else:
       X = X[:, 1]
       y = y[:, 2:]
           
    X = X.reshape(1, n_samples, 1).to(torch.float32)
    y = y.reshape(1, n_samples, 2).to(torch.float32)

    return X, y

model = DistributedAttention_twoway(
  x_items, 
  x_features, 
  embed_dim, 
  device, 
  rank, 
  output_dim=output_dim, # global
  heads=heads, 
  model_group=None
).to(torch.float32).to(device)

#optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

if rank == 0: 
   X = torch.tensor([[[0.7738, 0.2259],
         [0.5921, 0.5352],
         [0.9278, 0.0922],
         [0.2943, 0.3762]]]).to(device)
   y = torch.tensor([[[0.5278, 0.9356, 0.1470],
         [0.0444, 0.1539, 0.1956],
         [0.1278, 0.0125, 0.1721],
         [0.8640, 0.3388, 0.2108]]]).to(device)
else:
   X = torch.tensor([[[0.3826, 0.9717],
         [0.6686, 0.3781],
         [0.3625, 0.4136],
         [0.8099, 0.7743]]]).to(device)
   y = torch.tensor([[[0.2019, 0.7259, 0.4731],
         [0.8861, 0.7264, 0.8868],
         [0.5535, 0.4012, 0.1849],
         [0.7310, 0.3653, 0.8188]]]).to(device)
y = y.reshape(1, 4, 3)
   
result = model(X)
loss = torch.nn.MSELoss()(result, y)

loss.backward()

for param in model.parameters():
   if param.grad is not None:
      print(param.grad/2)
#print(result)
  
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)

for i in range(10):
  values = model(X)
  loss = torch.nn.MSELoss()(values, values*0.1)
  
  if rank == 0:
    print(loss)
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
  

torch.distributed.destroy_process_group()