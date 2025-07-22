import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

def gen_pointer_data(seq_len=10, dim=4):
    """
    Generate a sequence with random tokens and a one-hot pointer token at the end.
    Output is the token pointed to.

    Returns:
        x: (1, seq_len + 1, dim)
        y: (1, 1, dim)
    """
    x_tokens = torch.rand(seq_len, dim)
    index = torch.randint(0, seq_len, (1,))
    pointer_token = torch.zeros(dim)
    pointer_token[index.item() % dim] = 1.0  # one-hot index encoding

    x = torch.cat([x_tokens, pointer_token.unsqueeze(0)], dim=0)  # (seq_len+1, dim)
    y = x_tokens[index]  # target: token at the indexed position

    return x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)  # add batch dim

class PointerTransformer(nn.Module):
    def __init__(self, dim, seq_len):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.pos = nn.Parameter(torch.randn(1, seq_len + 1, dim))  # +1 for pointer token
        self.dim = dim

    def forward(self, x):
        x = x + self.pos
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        attn = (q @ k.transpose(-2, -1)) / (self.dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        return self.out_proj(out[:, -1:, :])  # predict from pointer token
    
model = PointerTransformer(dim=4, seq_len=10).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for i in range(50000):
    x, y = gen_pointer_data(seq_len=10, dim=4)
    pred = model(x)
    loss = F.mse_loss(pred, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if i % 100 == 0:
        print(f"{i}: loss = {loss.item():.6e}")
    if loss.item() < 1e-5:
        print(f"Converged at step {i}")
        break

