import torch
from torch import permute, reshape
from torch.nn import Conv2d, ConvTranspose2d
from utils import load_constant_mask_2d, getPositionEncoding
import Linear

class PatchEmbedding2D(torch.nn.Module):
  """2D Patch Embedding operation."""

  def __init__(self, patch_size, dim, device,  in_channels=72, padding=(0,0), stride=(8,8), xlat=721):
    """
    Initialize patch embedding operation.
    
    patch_size: Tuple(int, int)
        Number of pixels in (lat, lon) dimensions per patch
    dim: int
        Hidden dimension
    device: String
        Device that the operation is running on
    in_channels: int
        Total number of channels 
        equal to n_pressure_levels * n_pressure_fields + n_surface_fields + masks
        = 13 pressure levels * 5 pressure fields + 4 surface fields + 3 masks
    """
    super().__init__()
    self.patch_size = patch_size
    self.dim = dim
    # Here we use convolution to partition data into cubes
    # in_channels = 13 pressure levels x 5 fields + 4 variables + 3 masks
    # i.e., fields are (Z, Q, T, U, V)
    if padding[0] == patch_size and padding[1] == patch_size:
      padding = (0,0)
    self.stride = stride

    
    self.conv_surface = Conv2d(in_channels=in_channels, out_channels=dim, kernel_size=patch_size, stride=self.stride, padding=padding)

    # Load constant masks from the disc
    
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(patch_size, xlat=xlat)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
      
  def forward(self, input):
    """
    Forward pass of 2D patch embedding.
    
    input: Tensor
      of shape (n_batch,  n_fields*n_vert, n_lat, n_lon) 
      n_vert, n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (721 x 1440) with patch size of (8,8) -> (14, 728, 1440).
      i.e., in standard model, n_variables*n_vert = 5 vars * 13 pressure heights
    input_surface: Tensor
      of shape (n_batch, n_variables, n_lat, n_lon) 
      n_lat, n_lon are the number of pixels in the lat and lon resolution after
      padding, done in the dataloader step.
      i.e., (721 x 1440) with patch size of (8,8) -> (728, 1440)

    Returns
    -------
    x: Tensor
      of shape (n_batch, n_patch_lon*n_patch_lat, hidden_dim)
      i.e., for Lite models, (n_patch_lon, n_patch_lat) = (91, 180)

    """
    # Input should be padded already, according to the patch size
    input_shape = input.shape
    # Add three constant fields to the surface fields
    # Need to broadcast in this case because we are copying the data over more than 1 dimension
    # Broadcast to 4D data
    land_mask  = torch.broadcast_to(self.land_mask,   (input_shape[0], 1, input_shape[2], input_shape[3]))
    soil_type  = torch.broadcast_to(self.soil_type,   (input_shape[0], 1, input_shape[2], input_shape[3]))
    topography = torch.broadcast_to(self.topography,  (input_shape[0], 1, input_shape[2], input_shape[3]))
      
    input_surface = torch.cat((input, land_mask, soil_type, topography), dim=1)

    # Apply a linear projection for patch_size[1]*patch_size[2] patches
    # shape: (nData, fields, latitude, longitude)
    input_surface = self.conv_surface(input_surface)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, latitude, longitude, fields)
    x = permute(input_surface, (0, 2, 3, 1))
    # Dimensions: (nData,  latitude, longitude, fields)
    x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))
        
    return x
  

class PatchRecovery2D(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, dim, out_channels=69, padding=(0,0), stride=(8,8)):
    """
    2D Patch recovery.

    A transpose convolution operation is performed over the pressure and surface outputs to recover the forecasted fields.

    patch_size: Tuple(int, int, int)
        Number of pixels in (lat, lon) dimensions per patch
    dim: int
      Hidden dimension
    in_channels: int
      Total number of channels 
      equal to n_pressure_levels * n_pressure_fields + n_surface_fields
      = 13 pressure levels * 5 pressure fields + 4 surface fields = 69
    """
    super().__init__()
    # Here we use two transposed convolutions to recover data
    
    self.conv = ConvTranspose2d(in_channels=dim, out_channels=out_channels, kernel_size=patch_size, stride=stride, padding=padding)
    self.out_channels = out_channels
    self.patch_size = patch_size

  def forward(self, x, n_patch_lat, n_patch_lon, output_size):
    """
    2D inverse operation of the patch embedding operation.
    
    x: Tensor
      of shape (n_batch, n_patch_lat*n_patch_lon, 2*hidden_dim)
    n_patch_lat: int
      number of patches in the lat dimension
    n_patch_lon: int
      number of patches in the lon dimension

    Returns
    -------
    output: Tensor
      of shape (n_batch, n_levels * n_fields, n_lat, n_lon)
    output_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon)
    """
    # Reshape x back to three dimensions
    # Dimensions: (nData, pressure level * latitude * longitude, fields)
    x = permute(x, (0, 2, 1))
    # Dimensions: (nData, fields, pressure level, latitude, longitude)
    
    x = reshape(x, shape=(x.shape[0], x.shape[1], n_patch_lat, n_patch_lon))

    # Call the transposed convolution
    output = self.conv(x, output_size=output_size)
    output = output.permute(0, 2, 3, 1)

    # output shape: [n_batch, n_lat, n_lon, fields]
    return output
  
class DistributedPatchEmbedding(torch.nn.Module):
  def __init__(self, n_lat, n_lon, patch_size, var_in, embed_dim, comm, device, torch_rank):
    super().__init__()
    self.n_lat, self.n_lon = n_lat, n_lon
    self.patch_size = patch_size
    self.var_in = var_in
    self.embed_dim = embed_dim
    self.patch_embed = Linear.DistributedXWT(patch_size[0]*patch_size[1]*var_in//2, embed_dim//2, comm, device, torch_rank)
    self.device = device
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size)
    self.land_mask = self.land_mask
    self.soil_type = self.soil_type
    self.topography = self.topography
    self.torch_rank = torch_rank
    self.layernorm = torch.nn.LayerNorm(self.patch_size[0]*self.patch_size[1]*self.var_in)

  def forward(self, x):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    x_shape = x.shape
    land_mask  = torch.broadcast_to(self.land_mask,   (x_shape[0], 1, x_shape[2], x_shape[3]))
    soil_type  = torch.broadcast_to(self.soil_type,   (x_shape[0], 1, x_shape[2], x_shape[3]))
    topography = torch.broadcast_to(self.topography,  (x_shape[0], 1, x_shape[2], x_shape[3]))  

    x = torch.cat((x, land_mask, soil_type, topography), dim=1)
    
    # Rearrange x to have "windows"
    #print(f"x_shape in patch embed {x.shape}")
    # Each GPU holds: [n_batch, n_var, n_lat, n_lon]
    x_shape = x.shape
    
    
    # chatgpt 
    x_patches = torch.nn.functional.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
    x = x_patches.permute(0, 2, 3, 1, 4, 5).reshape(x_shape[0], -1, 72 * 16 * 16)

    #x = x.reshape(x_shape[0], x_shape[1], x_shape[2] // self.patch_size[0], self.patch_size[0], x_shape[3] // self.patch_size[1], self.patch_size[1])
    ## current x shape: (n_batch, vars, n_lat/p0, p0, n_lon/p1, p1)
    #x = x.permute(0, 2, 4, 3, 5, 1)
    ## current x shape: (n_batch, n_lat/p0, n_lon/p1, p0, p1, vars)
    #x_shape = x.shape
    #x = x_patches.reshape(x_shape[0], x_shape[1]*x_shape[2], self.patch_size[0]*self.patch_size[1]*self.var_in)
    fraction_1 = x.shape[-2] // 2
    fraction_2 = x.shape[-1] // 2
    #x = self.layernorm(x)
    # apply MLP/convolution
    if self.torch_rank == 0:
      x = self.patch_embed(x[:, :fraction_1,  :fraction_2].contiguous().to(self.device))
    elif self.torch_rank == 1:
      x = self.patch_embed(x[:, :fraction_1,  fraction_2:].contiguous().to(self.device))
    elif self.torch_rank == 2:
      x = self.patch_embed(x[:, fraction_1:,  :fraction_2].contiguous().to(self.device))
    elif self.torch_rank == 3:
      x = self.patch_embed(x[:, :fraction_1,  :fraction_2].contiguous().to(self.device))
    
    # x shape: (n_batch, n_lat/p0, n_lon/01, embed_dim)
    return x


class DistributedPatchRecovery(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, embed_dim, out_channels, comm, device, torch_rank):
    """
    2D Patch recovery.

    A transpose convolution operation is performed over the pressure and surface outputs to recover the forecasted fields.

    patch_size: Tuple(int, int, int)
        Number of pixels in (lat, lon) dimensions per patch
    dim: int
      Hidden dimension
    in_channels: int
      Total number of channels 
      equal to n_pressure_levels * n_pressure_fields + n_surface_fields
      = 13 pressure levels * 5 pressure fields + 4 surface fields = 69
    """
    super().__init__()
    # Here we use two transposed convolutions to recover data
    self.conv = Linear.DistributedXWT(embed_dim//2, patch_size[0]*patch_size[1]*out_channels, comm, device, torch_rank)
    self.patch_size = patch_size
    self.out_channels = out_channels

  def forward(self, x, n_patch_lat, n_patch_lon):
    """
    2D inverse operation of the patch embedding operation.
    
    x: Tensor
      of shape (n_batch, n_patch_lat*n_patch_lon, 2*hidden_dim)
    n_patch_lat: int
      number of patches in the lat dimension
    n_patch_lon: int
      number of patches in the lon dimension

    Returns
    -------
    output: Tensor
      of shape (n_batch, n_levels * n_fields, n_lat, n_lon)
    output_surface: Tensor
      of shape (n_batch, n_fields, n_lat, n_lon)
    """
    
    # Call the transposed convolution
    #print(f"1 - x shape {x.shape}")
    output = self.conv(x)

    #print(f"2 - conv shape {output.shape}")
    output_shape = output.shape
    # [4, n_patch_lat*n_patch_lon/2, patch_size_1*patch_size2*n_vars//2]
  


    # TODO

    output = output.view(output_shape[0], n_patch_lat, n_patch_lon//2, self.patch_size[0], self.patch_size[1], self.out_channels)
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    output_shape = output.shape
    output = output.view(output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]*output_shape[4], output_shape[5])

    # output shape: [n_batch, n_patch/2, n_var/2]
    return output