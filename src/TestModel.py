import torch
import PatchEmbedding
from Linear import MixerBlock, MixerBlockSequential
import Linear

class Model(torch.nn.Module):
    def __init__(self, device, comm, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        hidden_dimension = 1000
        self.patch_size = (4, 4)
        self.n_lat, self.n_lon = 720//self.patch_size[0], 1440//self.patch_size[1]
        self.embedding = PatchEmbedding.DistributedPatchEmbedding(720, 1440, self.patch_size, 72, hidden_dimension, comm, device, torch_rank)
        self.n_blocks = 2
        self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon//2, hidden_dimension//2, hidden_dimension//2, comm, device, torch_rank) for i in range(self.n_blocks)])
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension//2)
        if self.torch_rank == 3:
            self.num_variables = 18
        else:
            self.num_variables = 18
            
        self.layer_norm2 = torch.nn.LayerNorm(self.num_variables*2)    
        self.layer_norm3 = torch.nn.LayerNorm(36*2)    
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.recovery = PatchEmbedding.DistributedPatchRecovery(self.patch_size, hidden_dimension, 72, comm, device, torch_rank)
        self.linearlayer = Linear.DistributedXWT(self.num_variables*4, self.num_variables*2, comm, device, torch_rank)
        
        self.activation = torch.nn.GELU()
        self.device = device
        
        
    def forward(self, x):
        
        x_shape = x.shape
        x_res = x
        fraction = 72 // 4
        
        # x_res shape: [n_batch, n_var, n_lat, n_lon]
        x_res = x_res.permute(0, 2, 3, 1)
        # x_res shape after: [n_batch, n_lat, n_lon, n_var]
        
        x_res = x_res.reshape(x_res.shape[0], -1, x_res.shape[-1])
        x_res = torch.nn.functional.pad(x_res, (0,3)) 
                
        fraction_1 = x_res.shape[-2] // 2
        fraction_2 = 72 // 2
        if self.torch_rank == 0:
            x_res = x_res[:, :fraction_1,  :fraction_2].contiguous().to(self.device)
        elif self.torch_rank == 1:
            x_res = x_res[:, :fraction_1,  fraction_2:].contiguous().to(self.device)
        elif self.torch_rank == 2:
            x_res = x_res[:, fraction_1:,  fraction_2:].contiguous().to(self.device)
        elif self.torch_rank == 3:
            x_res = x_res[:, :fraction_1,  :fraction_2].contiguous().to(self.device)

        # x res for each GPU: [n_batch, n_lat, n_lon, n_var//4]
        # x_shape [n_batch, n_var, 720, 1440]
        x = self.embedding(x.to('cpu'))
        
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x.clone()

        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        
        x = self.layer_norm(x + x_res2)
        
        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x = self.recovery(x, self.n_lat, self.n_lon) 
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        
        x = self.layer_norm2(x) # layer norm must go before concat 
        
  #      print(f"{self.device} x shape {x.shape}, x_res shape {x_res.shape}")
        # TODO: START HERE
        x = torch.concat((x, x_res), dim=2)
        # After recovery: [n_batch, n_lat, n_lon, n_var]

        x = self.layer_norm3(x)
        #x = x.reshape(x.shape[0], 720*1440//2, 2*self.num_variables)
        
 #       print(f"x.shape before lin layer {x.shape}")
        x = self.linearlayer(x)
#        print(f"x.shape after lin layer {x.shape}")
        # [n_batch, ]


        

        #print(f"my rank is {self.torch_rank}; x_res shape {x_res.shape}; x shape {x.shape}")       
        # x shape: [n_batch, 720, 1440, 72//4]
        return x 
    
class ModelSequential(torch.nn.Module):
    def __init__(self, device, comm, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        hidden_dimension = 2000
        self.patch_size = (16, 16)
        self.n_lat, self.n_lon = 720//self.patch_size[0], 1440//self.patch_size[1]
        self.embedding = PatchEmbedding.PatchEmbedding2D(self.patch_size, hidden_dimension, device, 72)
        self.n_blocks = 2
        self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_lat*self.n_lon, hidden_dimension, hidden_dimension) for i in range(self.n_blocks)])
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        if self.torch_rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69
            
        self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)    
        self.layer_norm3 = torch.nn.LayerNorm(69*2)    
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, 69)
        self.linearlayer = torch.nn.Linear(self.num_variables*2, self.num_variables)
        
        self.activation = torch.nn.GELU()
        
        
    def forward(self, x):
        
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)
        
        
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)
        #x_res2 = self.layer_norm(x).clone()
        x_res2 = x.clone()
        # x shape: [1, 1152, 2000]
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        x = self.layer_norm(x + x_res2)
        x = self.recovery(x, self.n_lat, self.n_lon) 
        
        #x[:,:,:,:self.num_variables] + x_res
        x = self.layer_norm2(x) # layer norm must go before concat 
        x = torch.concat((x[:,:,:,:self.num_variables], x_res), dim=3)
        x = self.layer_norm3(x)
        x = x.reshape(x_shape[0], 720*1440, self.num_variables*2)
        x = self.linearlayer(x)
        x = x.reshape(x_shape[0], 720, 1440,self.num_variables)
        
        
        #x = self.layer_norm3(x)

        #print(f"my rank is {self.torch_rank}; x_res shape {x_res.shape}; x shape {x.shape}")       
        # x shape: [n_batch, 720, 1440, 72//4]
        return x 