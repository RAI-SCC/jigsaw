import torch
import PatchEmbedding
from Linear import MixerBlock

class Model(torch.nn.Module):
    def __init__(self, device, comm, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        hidden_dimension = 9000
        self.patch_size = (15, 15)
        self.n_lat, self.n_lon = 720//self.patch_size[0], 1440//self.patch_size[1]
        self.embedding = PatchEmbedding.DistributedPatchEmbedding(720, 1440, self.patch_size, 72, hidden_dimension, comm, device, torch_rank)
        self.n_blocks = 5
        self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon, hidden_dimension//4, hidden_dimension//4) for i in range(self.n_blocks)])
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.recovery = PatchEmbedding.DistributedPatchRecovery(self.patch_size, hidden_dimension, 72, comm, device, torch_rank)

        
    def forward(self, x):
        
        x_shape = x.shape
        x_res = x.clone()
        fraction = 72 // 4
        x_res = x_res.permute(0, 2, 3, 1)
        x_res = x_res[:, :, :, self.torch_rank*fraction:(self.torch_rank+1)*fraction]
        
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)
        # x shape: [1, 1152, 2000]
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        
        x = self.recovery(x, self.n_lat, self.n_lon) 
        
        if self.torch_rank == 3:
            x = x[:,:,:,:15]

        #print(f"my rank is {self.torch_rank}; x_res shape {x_res.shape}; x shape {x.shape}")       
        # x shape: [n_batch, 720, 1440, 72//4]
        return x + x_res