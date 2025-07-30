import Linear
import torch
from torch.nn import Softmax
import utils


class DistributedAttention_oneway(torch.nn.Module):
  def __init__(self, x_items, x_features, embed_dim, device, rank, model_group=None):
    super().__init__()
    self.model_group = model_group
    
    self.x_items = x_items
    self.embed_dim = embed_dim
    self.x_features = x_features
    
    self.q = torch.nn.Linear(x_features, embed_dim)
    self.k = torch.nn.Linear(x_features, embed_dim)
    self.v = torch.nn.Linear(x_features, embed_dim)
    
    self.softmax = Softmax(dim=-1)
    
    self.device = device
    self.rank = rank
    self.scale = embed_dim ** (-0.5) # should be (embed_dim/heads)

    self.linear = torch.nn.Linear(embed_dim, x_features)
    self.pos_embedding = torch.nn.Parameter(torch.randn(1, x_items, x_features))
    

  def forward(self, x, model_group=None):
    """
    x: (1, x_items, x_features)
    returns: same shape
    """
    x = x + self.pos_embedding  # optional but safe

    q = self.q(x)  # (1, x_items, embed_dim)
    k = self.k(x)
    v = self.v(x)

    q = self.scale * q
    attention_scores = torch.matmul(q, k.transpose(-2, -1))  # (1, x_items, x_items)

    attention_weights = self.softmax(attention_scores)  # along last dim
    # these weights should ideally be close to identity

    attended = torch.matmul(attention_weights, v)  # (1, x_items, embed_dim)

    out = self.linear(attended)  # (1, x_items, x_features)
    #print("Mean off-diagonal attention weight:",
    #  (attention_weights[0] - torch.eye(x_items, device=device)).abs().mean().item())
    return out
  
class MiniTransformer(torch.nn.Module):
    def __init__(self, dim: int, seq_len: int):
        super().__init__()
        self.qkv = torch.nn.Linear(dim, dim * 3)
        self.proj = torch.nn.Linear(dim, dim)
        self.pos = torch.nn.Parameter(torch.randn(1, seq_len, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos  # (1, seq_len, dim)
        qkv = self.qkv(x)  # (1, seq_len, 3*dim)
        q, k, v = qkv.chunk(3, dim=-1)

        scale = dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (1, seq, seq)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (1, seq_len, dim)
        return self.proj(out)  # (1, seq_len, dim)
    
device, _, rank, _ = utils.init_CUDA()

x_items, x_features, embed_dim = 20, 8, 32 # local values

#model = DistributedAttention_oneway(x_items, x_features, embed_dim, device, rank, model_group=None).to(device)

torch.manual_seed(7)

batch, seq_len, dim = 100, 20, x_features
# Learnable causal linear map: weights for each timestep

def gen_data(seq_len: int = 100, dim: int = 8):
    """
    Learnable but fixed per-token linear transformation:
    y[t] = x[t] @ W  (same W for all t)

    Returns:
        x: (1, seq_len, dim)
        y: (1, seq_len, dim)
    """
    x = torch.rand(1, seq_len, dim).to(device)
    y = x @ W_fixed  # broadcast matmul per token
    return x.to(device), y.to(device)

W_fixed = torch.randn(dim, dim).to(device)
model = MiniTransformer(dim, seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(100000):
  x, y = gen_data(seq_len=seq_len, dim=8)  
  values = model(x)
  
  loss = torch.nn.functional.mse_loss(values, y)
  if rank == 0:
    print(loss)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  
  


torch.distributed.destroy_process_group()