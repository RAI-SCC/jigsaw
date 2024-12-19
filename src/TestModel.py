import torch
import PatchEmbedding
from Linear import MixerBlock, MixerBlockSequential
import Linear
import utils

class Model(torch.nn.Module):
    def __init__(self, device, comm, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        hidden_dimension = 2000
        self.patch_size = (16, 16)
        self.n_lat, self.n_lon = 720//self.patch_size[0], 1440//self.patch_size[1]
        self.embedding = PatchEmbedding.DistributedPatchEmbedding(720, 1440, self.patch_size, 72, hidden_dimension, comm, device, torch_rank)
        self.n_blocks = 0
        self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon//2, hidden_dimension//2, hidden_dimension//2, comm, device, torch_rank) for i in range(self.n_blocks)])
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension//2)
        if self.torch_rank == 3:
            self.num_variables = 36
        else:
            self.num_variables = 36
            
        self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)    
        self.layer_norm3 = torch.nn.LayerNorm(36*2)    
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.recovery = PatchEmbedding.DistributedPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, comm, device, torch_rank)
        self.linearlayer = Linear.DistributedXWT(self.num_variables*2, self.num_variables*1, comm, device, torch_rank)
        
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
        
        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
        x = self.recovery(x, self.n_lat, self.n_lon) 
        
        
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        
        x = self.layer_norm2(x) # layer norm must go before concat 
        
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
    def __init__(self, config_dict, device, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        hidden_dimension = config_dict['model']['hidden_dim']
        self.patch_size = (config_dict['model']['patch_size'], config_dict['model']['patch_size'])
        self.stride = config_dict['model']['stride']
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

        padding = ((self.xlat+self.patch_size[0]-self.stride)%self.stride, (self.xlon+self.patch_size[1]-self.stride)%self.stride)
        
        self.n_lat = (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
        self.n_lon = (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
        
        self.embedding = PatchEmbedding.PatchEmbedding2D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat, n_patches=self.n_lat*self.n_lon)
        
        self.pos_encoding = config_dict['model']['positional_encoding']
    
        if self.pos_encoding:  
            self.positional_encoding = utils.getPositionEncoding(self.n_lat*self.n_lon, hidden_dimension, n=10000).to(device)
        
        self.n_blocks = config_dict['model']['mixing_blocks']
        self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_lat*self.n_lon, hidden_dimension, hidden_dimension) for i in range(self.n_blocks)])
        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        if self.torch_rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69
        
        self.add = config_dict['model']['add']

        if not self.add:    
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)    
            self.layer_norm3 = torch.nn.LayerNorm(69*2)    
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69)
        if not self.add:
            self.linearlayer = torch.nn.Linear(self.num_variables*2, self.num_variables)
        
        self.activation = torch.nn.GELU()

        if self.add:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm])
            }
            )
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        
    def forward(self, x):
        
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)
        
        
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x)
        # x: shape [1, n_patches, hidden_dim]
        if self.pos_encoding:
            x = x + self.positional_encoding # is this a good idea?
            
        #x_res2 = self.layer_norm(x).clone()
        x_res2 = x.clone()
        # x shape: [1, 1152, 2000]
        for i in range(self.n_blocks):
            x = self.blocks[i](x)

        x = self.layer_norm(x + x_res2)
        
        x = self.recovery(x, self.n_lat, self.n_lon, output_size=(x_res.shape[0], x_res.shape[3], x_res.shape[1], x_res.shape[2])) 
        
        if not self.add:
        #x[:,:,:,:self.num_variables] + x_res
            x = self.layer_norm2(x) # layer norm must go before concat 
        #
            x = torch.concat((x[:,:,:,:self.num_variables], x_res), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat*self.xlon, self.num_variables*2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon,self.num_variables)
            return x 
        else:
            x = x.reshape(x_shape[0], self.xlat, self.xlon,self.num_variables)
            return x + x_res