import torch
import PatchEmbedding
from Linear import MixerBlock, MixerBlockSequential, MixerBlock_twoway
import Linear
import utils
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, config_dict, device, rank, node_group):
        super().__init__()
        self.rank = rank
        self.device = device
        hidden_dimension = config_dict['model']['hidden_dim']        
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

        if config_dict['model']['patch_embed'] == 1:
            self.patch_size = config_dict['model']['patch_size']
            self.stride = config_dict['model']['stride']
            self.n_lat, self.n_lon = self.xlat, self.xlon//self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(self.xlat, self.xlon, self.patch_size, 72, hidden_dimension, device, rank, xlat=self.xlat, stride=self.stride)
        else:
            self.patch_size = (config_dict['model']['patch_size'], config_dict['model']['patch_size'])
            self.n_lat, self.n_lon = self.xlat//self.patch_size[0], self.xlon//self.patch_size[1]
            self.embedding = PatchEmbedding.DistributedPatchEmbedding(self.xlat, self.xlon, self.patch_size, 72, hidden_dimension,  device, rank, xlat=self.xlat)

        self.n_blocks = config_dict['model']['mixing_blocks']

        if self.stride < self.patch_size:
            self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon, hidden_dimension//2, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], device, rank, parallelism=config_dict['model_parallel']['parallelism']) for i in range(self.n_blocks)])
        else:
            self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon//2, hidden_dimension//2, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], device, rank, parallelism=config_dict['model_parallel']['parallelism']) for i in range(self.n_blocks)])
        self.layer_norm = Linear.DistributedLayerNorm(hidden_dimension//2, self.rank, self.device)
        
        if self.rank == 3:
            self.num_variables = 36
        else:
            self.num_variables = 36
            
        self.layer_norm2 = Linear.DistributedLayerNorm(self.num_variables, self.rank, self.device)    
        self.layer_norm3 = Linear.DistributedLayerNorm(36*2, self.rank, self.device)    
        self.concat = config_dict['model']['concat']
        self.linear_residual = config_dict['model']['linear_residual']
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict['model']['add']
        if config_dict['model']['patch_embed'] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, device, rank, concat=True, stride=self.stride)
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, device, rank, concat=False, stride=self.stride)
        else:
            self.recovery = PatchEmbedding.DistributedPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, device, rank)
        self.linearlayer = Linear.DistributedXWT_fourway(self.num_variables*2, self.num_variables*1, device, rank)
        
        self.activation = torch.nn.GELU()
        self.model_group = None
        
        if self.linear_residual:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm])
            }
            )
        
        
    def forward(self, x, rollout_=0):
        #print(f'I am rank {self.rank} and x has shape {x.shape}')
        x_res = x
        
        x_res = x_res.reshape(x_res.shape[0], x_res.shape[1]*x_res.shape[2], x_res.shape[3])
        
        # x res for each GPU: [n_batch, n_lat, n_lon, n_var//4]
        # x_shape [n_batch, n_var, 720, 1440]
        x = self.embedding(x, self.model_group)
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x
        
        for j in range(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x, self.model_group)
        
        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        if self.concat:
            x = self.layer_norm(x, self.rank, self.device)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2, self.rank, self.device)
        x = self.recovery(x, self.n_lat, self.n_lon, self.model_group) 
        
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        
        if self.add:
            return x + x_res
        
        if self.linear_residual:
            x = self.layer_norm2(x, self.rank, self.device)
            x = torch.concat((x, x_res), dim=2)

            ### After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x, self.rank, self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = self.linearlayer(x, self.model_group)
            return x  
        else:
            return x
        
class Model_twoway(torch.nn.Module):
    def __init__(self, config_dict, device, rank, group):
        super().__init__()
        self.rank = rank
        self.device = device
        hidden_dimension = config_dict['model']['hidden_dim']        
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']
        self.group = group # Local groups [0,1], [2,3]...
        
        if config_dict['model']['patch_embed'] == 1:
            self.patch_size = config_dict['model']['patch_size']
            self.stride = config_dict['model']['stride']
            self.n_lat, self.n_lon = self.xlat, self.xlon//self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding_twoway(self.xlat, self.xlon, self.patch_size, 72, hidden_dimension,  device, rank, xlat=self.xlat, stride=self.stride)
        self.n_blocks = config_dict['model']['mixing_blocks']
        if self.stride < self.patch_size:
            self.blocks = torch.nn.ModuleList([MixerBlock_twoway(self.n_lat*self.n_lon*2, hidden_dimension, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], device, rank) for i in range(self.n_blocks)])
        else:
            self.blocks = torch.nn.ModuleList([MixerBlock_twoway(self.n_lat*self.n_lon, hidden_dimension, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], device, rank) for i in range(self.n_blocks)])
        self.layer_norm = Linear.torch.nn.LayerNorm(hidden_dimension//2)#, self.rank, self.device)
        
        if self.rank == 3:
            self.num_variables = 72
        else:
            self.num_variables = 72
            
        self.layer_norm2 = torch.nn.LayerNorm(self.num_variables//2)    
        self.layer_norm3 = torch.nn.LayerNorm(36*2)

        self.concat = config_dict['model']['concat']
        self.linear_residual = config_dict['model']['linear_residual']
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict['model']['add']
        if config_dict['model']['patch_embed'] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery_twoway(self.patch_size, hidden_dimension, self.num_variables, device, rank, concat=True, stride=self.stride)
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery_twoway(self.patch_size, hidden_dimension, self.num_variables,  device, rank, concat=False, stride=self.stride)
        
        self.linearlayer = Linear.DistributedXWT_twoway(self.num_variables, self.num_variables*1, device, rank)
        
        self.activation = torch.nn.GELU()

        if self.linear_residual:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm])
            }
            )
        self.model_group = group
        
        
    def forward(self, x, rollout_=0):
        #print(f'I am rank {self.rank} and x has shape {x.shape}')
        x_res = x#.clone() 
        x_res = x_res.flatten(start_dim=1, end_dim=2)
        
        # x res for each GPU: [n_batch, n_lat, n_lon, n_var//4]
        # x_shape [n_batch, n_var, 720, 1440]
        x = self.embedding(x, self.model_group)
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x#.clone()
        
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x, self.model_group)
        
        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)
        
        x = self.recovery(x, self.n_lat, self.n_lon, self.model_group) 
        
        
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        
        if self.add:
            return x + x_res
        if self.linear_residual:
            x = self.layer_norm2(x)
            x = torch.concat((x, x_res), dim=2)

            ### After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            
            x = self.linearlayer(x, self.model_group)
            
            return x  
        else:
            return x
    
class ModelSequential(torch.nn.Module):
    def __init__(self, config_dict, device, rank):
        super().__init__()
        self.rank = rank
        hidden_dimension = config_dict['model']['hidden_dim']
        
        self.stride = config_dict['model']['stride']
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

        #padding = ((self.xlat+self.patch_size[0]-self.stride)%self.stride, (self.xlon+self.patch_size[1]-self.stride)%self.stride)
        padding = (0, 0)
        
        self.patch_embed = config_dict['model']['patch_embed']
        if self.patch_embed == 1:
            self.patch_size = config_dict['model']['patch_size']
            self.stride = config_dict['model']['stride']
            self.n_patches = (self.xlat*self.xlon-1*(self.patch_size-1)-1)//self.stride + 1
            
            self.dropout = config_dict['model']['dropout']
            self.embedding = PatchEmbedding.PatchEmbedding1D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_patches, hidden_dimension, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], self.dropout) for i in range(self.n_blocks)])
        elif self.patch_embed == 2:    
            self.patch_size = (config_dict['model']['patch_size'], config_dict['model']['patch_size'])
            self.embedding = PatchEmbedding.PatchEmbedding2D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.n_lat = (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
            self.n_lon = (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_lat*self.n_lon, hidden_dimension, hidden_dimension, self.dropout) for i in range(self.n_blocks)])
        else:
            self.patch_size = config_dict['model']['patch_size']
            self.n_lat = 720 // 1 # (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
            self.n_lon = 1440 // self.patch_size # (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
            self.dropout = config_dict['model']['dropout']
            self.embedding = PatchEmbedding.PatchEmbeddingTest(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(720*1440//self.patch_size, hidden_dimension, hidden_dimension, self.dropout) for i in range(self.n_blocks)])
        self.pos_encoding = config_dict['model']['positional_encoding']
    
        if self.pos_encoding:  
            self.positional_encoding = utils.getPositionEncoding(self.n_lat*self.n_lon, hidden_dimension, n=10000).to(device)
        
        self.latent_rollout = config_dict['model']['latent_rollout']

        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        if self.rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69
        
        self.add = config_dict['model']['add']
        self.concat = config_dict['model']['concat']
        self.linear_residual = config_dict['model']['linear_residual']
        if self.linear_residual:    
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)    
            self.layer_norm3 = torch.nn.LayerNorm(69*2)    
            self.linearlayer = torch.nn.Linear(self.num_variables*2, self.num_variables)
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        if self.patch_embed == 1:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery1D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=True, xlat=self.xlat, xlon=self.xlon)
            else:
                self.recovery = PatchEmbedding.PatchRecovery1D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False, xlat=self.xlat, xlon=self.xlon)
        elif self.patch_embed == 2:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=True)
            else:
                self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False)
        else: 
            self.recovery = PatchEmbedding.PatchRecoveryTest(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False)
        
        
        self.activation = torch.nn.GELU()
  
        if self.linear_residual:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm])
            }
            )
        
    def forward(self, x, rollout_=1):
        
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)
        
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x) # x: latent_t
        
        # x: shape [1, n_patches, hidden_dim]
        if self.pos_encoding:
            x = x + self.positional_encoding # is this a good idea?
            
        #x_res2 = self.layer_norm(x).clone()
        x_res2 = x.clone()
        
        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)   # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)
        
        if self.patch_embed == 2:
            x = self.recovery(x, self.n_lat, self.n_lon, output_size=(x_res.shape[0], x_res.shape[3], x_res.shape[1], x_res.shape[2])) 
        else: # 1D convolution
            x = self.recovery(x)
        

        if self.add:
            x = x.reshape(x_shape[0], self.xlat, self.xlon,self.num_variables)
            
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x) # layer norm must go before concat 
            x = torch.concat((x[:,:,:,:self.num_variables], x_res), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat*self.xlon, self.num_variables*2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x
            

class ModelInterpret(torch.nn.Module):
    def __init__(self, config_dict, device, rank):
        super().__init__()
        self.rank = rank
        hidden_dimension = config_dict['model']['hidden_dim']
        
        self.stride = config_dict['model']['stride']
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

        #padding = ((self.xlat+self.patch_size[0]-self.stride)%self.stride, (self.xlon+self.patch_size[1]-self.stride)%self.stride)
        padding = (0, 0)
        
        self.patch_embed = config_dict['model']['patch_embed']
        if self.patch_embed == 1:
            self.patch_size = config_dict['model']['patch_size']
            self.stride = config_dict['model']['stride']
            self.n_patches = (self.xlat*self.xlon-1*(self.patch_size-1)-1)//self.stride + 1
            
            self.dropout = config_dict['model']['dropout']
            self.embedding = PatchEmbedding.PatchEmbedding1D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_patches, hidden_dimension, config_dict['model']['spatial_hidden_dim_fraction'], config_dict['model']['features_hidden_dim_fraction'], self.dropout) for i in range(self.n_blocks)])
        elif self.patch_embed == 2:    
            self.patch_size = (config_dict['model']['patch_size'], config_dict['model']['patch_size'])
            self.embedding = PatchEmbedding.PatchEmbedding2D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.n_lat = (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
            self.n_lon = (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_lat*self.n_lon, hidden_dimension, hidden_dimension, self.dropout) for i in range(self.n_blocks)])
        else:
            self.patch_size = config_dict['model']['patch_size']
            self.n_lat = 720 // 1 # (self.xlat+2*padding[0]-1*(self.patch_size[0]-1)-1)//self.stride+1
            self.n_lon = 1440 // self.patch_size # (self.xlon+2*padding[1]-1*(self.patch_size[1]-1)-1)//self.stride+1
            self.dropout = config_dict['model']['dropout']
            self.embedding = PatchEmbedding.PatchEmbeddingTest(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
            self.n_blocks = config_dict['model']['mixing_blocks']
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(720*1440//self.patch_size, hidden_dimension, hidden_dimension, self.dropout) for i in range(self.n_blocks)])
        self.pos_encoding = config_dict['model']['positional_encoding']
    
        if self.pos_encoding:  
            self.positional_encoding = utils.getPositionEncoding(self.n_lat*self.n_lon, hidden_dimension, n=10000).to(device)
        
        self.latent_rollout = config_dict['model']['latent_rollout']

        self.layer_norm = torch.nn.LayerNorm(hidden_dimension)
        if self.rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69
        
        self.add = config_dict['model']['add']
        self.concat = config_dict['model']['concat']
        self.linear_residual = config_dict['model']['linear_residual']
        if self.linear_residual:    
            self.layer_norm2 = torch.nn.LayerNorm(self.num_variables)    
            self.layer_norm3 = torch.nn.LayerNorm(69*2)    
            self.linearlayer = torch.nn.Linear(self.num_variables*2, self.num_variables)
        
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        if self.patch_embed == 1:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery1D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=True, xlat=self.xlat, xlon=self.xlon)
            else:
                self.recovery = PatchEmbedding.PatchRecovery1D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False, xlat=self.xlat, xlon=self.xlon)
        elif self.patch_embed == 2:
            if self.concat:
                self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=True)
            else:
                self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False)
        else: 
            self.recovery = PatchEmbedding.PatchRecoveryTest(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69, concat=False)
        
        
        self.activation = torch.nn.GELU()
  
        if self.linear_residual:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm])
            }
            )
        
    def forward(self, x, rollout_=1):
        
        x_shape = x.shape
        x_res = x.clone()
        x_res = x_res.permute(0, 2, 3, 1)
        
        # x shape: [1, 69, 720, 1440]
        x = self.embedding(x) # x: latent_t
        
        # x: shape [1, n_patches, hidden_dim]
        if self.pos_encoding:
            x = x + self.positional_encoding # is this a good idea?
            
        #x_res2 = self.layer_norm(x).clone()
        x_res2 = x.clone()
        
        # x shape: [1, 1152, 2000]
        for rollout in torch.arange(rollout_):
            for i in range(self.n_blocks):
                x = self.blocks[i](x)   # x: latent_tplus1

        if self.concat:
            x = self.layer_norm(x)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2)
        
        if self.patch_embed == 2:
            x = self.recovery(x, self.n_lat, self.n_lon, output_size=(x_res.shape[0], x_res.shape[3], x_res.shape[1], x_res.shape[2])) 
        else: # 1D convolution
            x = self.recovery(x)
        

        if self.add:
            x = x.reshape(x_shape[0], self.xlat, self.xlon,self.num_variables)
            
            return x + x_res

        if self.linear_residual:
            x = self.layer_norm2(x) # layer norm must go before concat 
            x = torch.concat((x[:,:,:,:self.num_variables], x_res), dim=3)
            x = self.layer_norm3(x)
            x = x.reshape(x_shape[0], self.xlat*self.xlon, self.num_variables*2)
            x = self.linearlayer(x)
            x = x.reshape(x_shape[0], self.xlat, self.xlon, self.num_variables)
        return x