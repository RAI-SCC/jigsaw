import torch
import PatchEmbedding
from Linear import MixerBlock, MixerBlockSequential
import Linear
import utils
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, config_dict, device, comm, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
        self.device = device
        hidden_dimension = config_dict['model']['hidden_dim']        
        self.xlat, self.xlon = config_dict['data']['xlat'], config_dict['data']['xlon']

        if config_dict['model']['patch_embed'] == 1:
            self.patch_size = config_dict['model']['patch_size']
            self.stride = config_dict['model']['stride']
            self.n_lat, self.n_lon = self.xlat, self.xlon//self.patch_size
            self.embedding = PatchEmbedding.Distributed1DPatchEmbedding(self.xlat, self.xlon, self.patch_size, 72, hidden_dimension, comm, device, torch_rank, xlat=self.xlat)
        else:
            self.patch_size = (config_dict['model']['patch_size'], config_dict['model']['patch_size'])
            self.n_lat, self.n_lon = self.xlat//self.patch_size[0], self.xlon//self.patch_size[1]
            self.embedding = PatchEmbedding.DistributedPatchEmbedding(self.xlat, self.xlon, self.patch_size, 72, hidden_dimension, comm, device, torch_rank, xlat=self.xlat)
        self.n_blocks = config_dict['model']['mixing_blocks']
        self.blocks = torch.nn.ModuleList([MixerBlock(self.n_lat*self.n_lon//2, hidden_dimension//2, hidden_dimension//2, comm, device, torch_rank) for i in range(self.n_blocks)])
        self.layer_norm = Linear.DistributedLayerNorm(hidden_dimension//2, self.torch_rank, self.device)
        
        if self.torch_rank == 3:
            self.num_variables = 36
        else:
            self.num_variables = 36
            
        self.layer_norm2 = Linear.DistributedLayerNorm(self.num_variables, self.torch_rank, self.device)    
        self.layer_norm3 = Linear.DistributedLayerNorm(36*2, self.torch_rank, self.device)    
        self.concat = config_dict['model']['concat']
        self.linear_residual = config_dict['model']['linear_residual']
        # 72 = 69 ROUNDED TO NEAREST MULTLIPLE OF 4
        self.add = config_dict['model']['add']
        if config_dict['model']['patch_embed'] == 1:
            if self.concat:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, comm, device, torch_rank, concat=True)
            else:
                self.recovery = PatchEmbedding.Distributed1DPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, comm, device, torch_rank, concat=False)
        else:
            self.recovery = PatchEmbedding.DistributedPatchRecovery(self.patch_size, hidden_dimension, self.num_variables, comm, device, torch_rank)
        self.linearlayer = Linear.DistributedXWT(self.num_variables*2, self.num_variables*1, comm, device, torch_rank)
        
        self.activation = torch.nn.GELU()
        
        
    def forward(self, x, rollout_=0):
        #print(f'I am rank {self.torch_rank} and x has shape {x.shape}')
        x_res = x.clone() 
        
        x_res = x_res.flatten(start_dim=1, end_dim=2)
        
        # x res for each GPU: [n_batch, n_lat, n_lon, n_var//4]
        # x_shape [n_batch, n_var, 720, 1440]
        
        #x = x.to(self.device)
        
        x = self.embedding(x)
        # After embedding: x_shape [n_batch, n_patch_lat * n_patch_lon, embed_dim]
        x_res2 = x.clone()
        
        for i in range(self.n_blocks):
            x = self.blocks[i](x)
        

        # After MLP-mixing blocks: [n_batch, n_patch_lat * n_patch_lon/2, embed_dim/2]
        # After recovery: [n_batch, n_lat*n_lon//2, n_var//2]
        if self.concat:
            x = self.layer_norm(x, self.torch_rank, self.device)
            x = torch.concat((x, x_res2), dim=2)
        else:
            x = self.layer_norm(x + x_res2, self.torch_rank, self.device)
        
        x = self.recovery(x, self.n_lat, self.n_lon) 
        
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        
        if self.add:
            return x + x_res
        if self.linear_residual:
            x = torch.concat((x, x_res), dim=2)

            ### After recovery: [n_batch, n_lat, n_lon, n_var]
            x = self.layer_norm3(x, self.torch_rank, self.device)
            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = self.linearlayer(x)
            return x  
        else:
            return x
    
class ModelSequential(torch.nn.Module):
    def __init__(self, config_dict, device, torch_rank):
        super().__init__()
        self.torch_rank = torch_rank
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
            self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_patches, hidden_dimension, hidden_dimension, self.dropout) for i in range(self.n_blocks)])
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
        if self.torch_rank == 3:
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
            
class ModelUPT(torch.nn.Module):
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
        
        self.embedding = PatchEmbedding.PatchEmbedding2D(self.patch_size, hidden_dimension, device=device,in_channels=72, padding=padding, stride=self.stride, xlat=self.xlat)
        
        self.pos_encoding = config_dict['model']['positional_encoding']
    
        if self.pos_encoding:  
            self.positional_encoding = utils.getPositionEncoding(self.n_lat*self.n_lon, hidden_dimension, n=10000).to(device)
        
        self.latent_rollout = config_dict['model']['latent_rollout']

        self.n_blocks = config_dict['model']['mixing_blocks']
        self.blocks = torch.nn.ModuleList([MixerBlockSequential(self.n_lat*self.n_lon, hidden_dimension, hidden_dimension) for i in range(self.n_blocks)])
        if self.torch_rank == 3:
            self.num_variables = 69
        else:
            self.num_variables = 69
        
        self.add = config_dict['model']['add']

        self.recovery = PatchEmbedding.PatchRecovery2D(self.patch_size, hidden_dimension, padding=padding, stride=self.stride, out_channels=69)
        
        self.activation = torch.nn.GELU()

        if self.add:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks])
            }
            )
        else:
            self.params = torch.nn.ModuleDict({
            'embedding': torch.nn.ModuleList([self.embedding]),
            'recovery': torch.nn.ModuleList([self.recovery]),
            'common': torch.nn.ModuleList([self.blocks, self.layer_norm, self.layer_norm2, self.layer_norm3, self.linearlayer])
            }
            )       
        
    def forward(self, x, rollout_=1):
        
        x_shape = x.shape
        x_res = x.clone()

        x_res = x_res.permute(0, 2, 3, 1)
        
        # x shape: [1, 69, 720, 1440]
        latent_t = self.embedding(x) # x: latent_t
        # x: shape [1, n_patches, hidden_dim]
        if self.pos_encoding:
            latent_t = latent_t + self.positional_encoding # is this a good idea?
                
        x_res2 = latent_t.clone()

        # x shape: [1, 1152, 2000]
        
        for i in range(self.n_blocks):
            latent_t = self.blocks[i](latent_t)   # x: latent_tplus1
        
        latent_t = x_res2 + latent_t

        latent_tplus1 = latent_t.clone()
      
        decoded_t = self.recovery(x_res2, self.n_lat, self.n_lon, output_size=(x_res.shape[0], x_res.shape[3], x_res.shape[1], x_res.shape[2]))  # decoded_t

        decoded_tplus1 = self.recovery(latent_t, self.n_lat, self.n_lon, output_size=(x_res.shape[0], x_res.shape[3], x_res.shape[1], x_res.shape[2]))   # decoded_tplus1
        decoded_tplus1 = decoded_tplus1.reshape(x_shape[0], self.xlat, self.xlon,self.num_variables) # No adding; decoded_tplus1
        inverse_encoded = self.embedding(decoded_tplus1.permute(0, 3, 1, 2)) # compare x_encoder with x_approximator_out # inverse_encoded
                
        return decoded_tplus1, decoded_t.permute(0, 3, 1, 2), inverse_encoded, latent_tplus1
