import torch
from torch import permute, reshape
from torch.nn import Conv2d, ConvTranspose2d, Softmax

import Linear
from Dataset import load_constant_mask_2d


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

  def __init__(self, patch_size, dim, out_channels=69, padding=(0,0), stride=(8,8), concat=False):
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
    if concat:
      self.conv = ConvTranspose2d(in_channels=dim*2, out_channels=out_channels, kernel_size=patch_size, stride=stride, padding=padding)
    else:
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
  
class PatchEmbedding1D(torch.nn.Module):
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
    self.patch_size = patch_size#patch_size[0]*patch_size[1] # sq
    self.dim = dim#patch_size[0]*patch_size[1]*in_channels
    # Here we use convolution to partition data into cubes
    # in_channels = 13 pressure levels x 5 fields + 4 variables + 3 masks
    # i.e., fields are (Z, Q, T, U, V)
    #if padding[0] == patch_size and padding[1] == patch_size:
    #  padding = (0,0)
    self.stride = stride

    self.conv_surface = torch.nn.Conv1d(in_channels, out_channels=self.dim, kernel_size=self.patch_size, stride=self.stride)
    
    # Load constant masks from the disc
    
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=xlat)
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
    input_surface = input_surface.reshape(input_shape[0], input_shape[1]+3, -1)
    
    input_surface = self.conv_surface(input_surface)
    
    # Reshape x for calculation of linear projections
    # Dimensions: (nData, latitude, longitude, fields)
    x = permute(input_surface, (0, 2, 1))
    ## Dimensions: (nData,  latitude, longitude, fields)
    #x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))
        
    return x
  

class PatchRecovery1D(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, dim, out_channels=69, padding=(0,0), stride=(8,8), concat=False, xlat=721, xlon=1440):
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
    self.patch_size = patch_size
    self.stride = patch_size

    if concat:
      self.conv = torch.nn.ConvTranspose1d(in_channels=dim*2, out_channels=69, kernel_size=self.patch_size, stride=self.stride)
    else:
      self.conv = torch.nn.ConvTranspose1d(in_channels=dim, out_channels=69, kernel_size=self.patch_size, stride=self.stride)
    self.out_channels = out_channels
    self.patch_size = patch_size
    self.xlat, self.xlon = xlat, xlon

  def forward(self, x):
    """
    1D inverse operation of the patch embedding operation.
    
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
    # Dimensions: (nData, fields, lat*lon)
    
    # Call the transposed convolution
    output = self.conv(x)
    output = output.permute(0, 2, 1).reshape(x.shape[0], self.xlat, self.xlon, -1)

    # output shape: [n_batch, n_lat, n_lon, fields]
    return output


class PatchEmbeddingTest(torch.nn.Module):
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
    self.patch_size = patch_size#patch_size[0]*patch_size[1] # sq
    self.dim = self.patch_size*in_channels#patch_size[0]*patch_size[1]*in_channels
    # Here we use convolution to partition data into cubes
    # in_channels = 13 pressure levels x 5 fields + 4 variables + 3 masks
    # i.e., fields are (Z, Q, T, U, V)
    #if padding[0] == patch_size and padding[1] == patch_size:
    #  padding = (0,0)
    self.stride = stride

    self.conv_surface = torch.nn.Linear(self.dim, dim)
    
    # Load constant masks from the disc
    
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=xlat)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
      
  def forward(self, input):
    """
    Forward pass of 1D patch embedding.
    
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
    #input_surface = input_surface.reshape(input_shape[0], input_shape[1]+3, -1)

    n_batch, n_var, n_x, n_y  = input_surface.shape
    input_surface = input_surface.permute(0, 2, 3, 1) # variable @ end
    x_patches = (input_surface.reshape(n_batch, n_x, n_y//self.patch_size, self.patch_size, n_var)
                              .reshape(n_batch, n_x*n_y//self.patch_size, self.patch_size*n_var))


    x = self.conv_surface(x_patches)

    # Reshape x for calculation of linear projections
    # Dimensions: (nData, latitude, longitude, fields)
    
    ## Dimensions: (nData,  latitude, longitude, fields)
    #x = reshape(x, shape=(x.shape[0], x.shape[1]*x.shape[2], x.shape[-1]))
        
    return x
  

class PatchRecoveryTest(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, dim, out_channels=69, padding=(0,0), stride=(8,8), concat=False):
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
    self.patch_size = patch_size
    self.stride = stride
    self.conv = torch.nn.Linear(dim, patch_size*out_channels)
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
    
    
    # Dimensions: (nData, fields, lat*lon)
    
    # Call the transposed convolution
    output = self.conv(x)
    output = (output.view(output.shape[0], n_patch_lat, n_patch_lon, self.patch_size, self.out_channels)
                    .view(output.shape[0], 720, 1440, self.out_channels))
    

    # output shape: [n_batch, n_lat, n_lon, fields]
    return output

class DistributedPatchEmbedding(torch.nn.Module):
  def __init__(self, n_lat, n_lon, patch_size, var_in, embed_dim, comm, rank, device, xlat=720):
    super().__init__()
    self.n_lat, self.n_lon = n_lat, n_lon
    self.patch_size = patch_size
    self.var_in = var_in
    self.embed_dim = embed_dim
    self.patch_embed = Linear.DistributedXWT_fourway(patch_size[0]*patch_size[1]*var_in//2, embed_dim//2, comm, rank, device, n_channels=var_in, init_method='conv')
    self.device = device
    self.land_mask, self.soil_type, self.topography = load_constant_mask_2d(self.patch_size, xlat=720)
    self.land_mask = self.land_mask.to(device)
    self.soil_type = self.soil_type.to(device)
    self.topography = self.topography.to(device)
    self.rank = rank
    #self.layernorm = torch.nn.LayerNorm(self.patch_size[0]*self.patch_size[1]*self.var_in)

  def forward(self, x, model_group=None):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    x_shape = x.shape
    #land_mask  = torch.broadcast_to(self.land_mask,   (x_shape[0], 1, x_shape[2], x_shape[3]))
    #soil_type  = torch.broadcast_to(self.soil_type,   (x_shape[0], 1, x_shape[2], x_shape[3]))
    #topography = torch.broadcast_to(self.topography,  (x_shape[0], 1, x_shape[2], x_shape[3]))  
    x = x.permute(0, 2, 3, 1)
    #x = torch.cat((x, land_mask, soil_type, topography), dim=1).permute(0, 2, 3, 1)
    
    x_shape = x.shape
    
    n_batch, n_x, n_y, n_var = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
    
    fraction_1 = x.shape[2]//2 # num_patches_lon
    fraction_2 = x.shape[3]//2 # num_variables

    if self.rank == 0:
      x_local = x[:, :, :fraction_1,  :fraction_2]
    elif self.rank == 1:
      x_local = x[:, :, :fraction_1,  fraction_2:]
    elif self.rank == 2:
      x_local = x[:, :, fraction_1:, :fraction_2]
    elif self.rank == 3:
      x_local = x[:, :, fraction_1:, fraction_2:]
#
    n_x = x_local.shape[1]
    n_y = x_local.shape[2]

    x_patches_local = (x_local.reshape(n_batch, x_local.shape[1] // self.patch_size[0], self.patch_size[0], x_local.shape[2] // self.patch_size[1], self.patch_size[1], x_local.shape[3])
                       .permute(0, 1, 3, 2, 4, 5)) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, size_patch, n_var]
    
    x_patches_local = x_patches_local.reshape(n_batch, n_x // self.patch_size[0] * n_y // self.patch_size[1], self.patch_size[0] * self.patch_size[1] * fraction_2)
    x_patches_local = x_patches_local.contiguous().to(self.device)
    
    x = self.patch_embed(x_patches_local, model_group)
  
    # x shape: (n_batch, n_lat/p0, n_lon/01, embed_dim)
    return x

class DistributedPatchRecovery(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, embed_dim, out_channels, comm, rank, device, n_parallel, model_group):
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
    self.conv = Linear.DistributedXWT(embed_dim//2, patch_size[0]*patch_size[1]*out_channels, comm, rank, device, n_parallel, model_group, n_channels=embed_dim, init_method='conv')
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

    output = self.conv(x) # distributed 
    
    #print(f"2 - conv shape {output.shape}")
    output_shape = output.shape
    
    # [4, n_patch_lat*n_patch_lon/2, patch_size_1*patch_size2*n_vars//2]
    output = output.view(output_shape[0], n_patch_lat, n_patch_lon//2, self.patch_size[0], self.patch_size[1], self.out_channels)
    output = output.permute(0, 1, 3, 2, 4, 5).contiguous()
    output_shape = output.shape
    output = output.view(output_shape[0], output_shape[1]*output_shape[2], output_shape[3]*output_shape[4], output_shape[5])

    # output shape: [n_batch, n_patch/2, n_var/2]
    return output

class Distributed1DPatchEmbedding(torch.nn.Module):
  def __init__(self, n_lat, n_lon, patch_size, var_in, embed_dim, rank, device, n_parallel, model_group):
    super().__init__()

    self.n_lat, self.n_lon = n_lat, n_lon
    self.patch_size = patch_size
    self.var_in = var_in
    self.embed_dim = embed_dim

    if n_parallel == 2:
      self.patch_embed = Linear.DistributedXWT(patch_size*var_in//2, embed_dim,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='conv')
    elif n_parallel == 4:
      self.patch_embed = Linear.DistributedXWT(patch_size*var_in//2, embed_dim//2,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='conv')
    elif n_parallel == 8:
      self.patch_embed = Linear.DistributedXWT(patch_size*var_in//4, embed_dim//2,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='conv')
    
    self.rank = rank
    self.device = device
    
  def forward(self, x):
    """x: (n_batch, vars, n_lat, n_lon)"""
    # Load constant masks from the disc
    
    n_batch, n_x, n_y, n_var = x.shape
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(n_batch, n_x, n_y // self.patch_size, self.patch_size, n_var) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
    x_patches_local = x_patches_local.flatten(start_dim=1, end_dim=2)
    x_patches_local = x_patches_local.flatten(start_dim=2, end_dim=3)
    
    x = self.patch_embed(x_patches_local)
    
    return x


class Distributed1DPatchRecovery(torch.nn.Module):
  """2D Patch recovery option."""

  def __init__(self, patch_size, embed_dim, out_channels, rank, device, n_parallel, model_group, concat=False, group=None, stride=None):
    """
    1D Patch recovery.
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
    self.rank = rank
    self.device = device
    self.n_parallel = n_parallel
    self.model_group = model_group
    self.concat = concat

    if self.concat:
      self.conv = Linear.DistributedXWT(embed_dim, patch_size*out_channels, self.rank, self.device, self.n_parallel, self.model_group, n_channels=embed_dim, init_method='conv')
    else:
      self.conv = Linear.DistributedXWT(embed_dim//2, patch_size*out_channels, self.rank, self.device, self.n_parallel, self.model_group, n_channels=embed_dim, init_method='conv')
    self.patch_size = patch_size

    if self.n_parallel == 2:
      self.out_channels = out_channels // 2    
    else:
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
    output = self.conv(x) # distributed convolution
    output = output.view(output.shape[0], n_patch_lat, n_patch_lon, self.patch_size, self.out_channels)
    output = output.flatten(start_dim=2, end_dim=3)
    
    # output shape: [n_batch, n_patch/2, n_var/2]
    return output
  

# Transformer-based patch embedding
class DistributedAttention_twoway(torch.nn.Module):
  def __init__(self, n_lat, n_lon, patch_size, var_in, embed_dim, rank, device, n_parallel, model_group, xlat=720, stride=None):
    super().__init__()
    self.model_group = model_group

    self.n_lat, self.n_lon = n_lat, n_lon
    self.patch_size = patch_size
    self.var_in = var_in
    self.embed_dim = embed_dim
    if stride is None:
      self.stride=patch_size
    else:
      self.stride=stride

    self.q = Linear.DistributedXWT_twoway(patch_size*var_in//2, embed_dim,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='linear')
    self.k = Linear.DistributedXWT_twoway(patch_size*var_in//2, embed_dim,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='linear')
    self.v = Linear.DistributedXWT_twoway(patch_size*var_in//2, embed_dim,  rank, device, n_parallel, model_group, n_channels=var_in, init_method='linear')
    
    self.softmax = Linear.DistributedSoftMax(dim=-1, group=self.model_group)
    self.xwt = Linear.DistributedXWT_attention_twoway(rank, device, n_parallel, model_group) # model group TODO 
    
    self.device = device
    self.rank = rank
    self.scale = embed_dim ** -0.5 # should be (embed_dim/heads)
    self.Softmax = Softmax(dim=-1)

  def forward(self, x, model_group):
    """
    x: (n_batch, vars, n_lat, n_lon)
    """
    # Load constant masks from the disc
    
    n_batch, n_x, n_y, n_var = x.shape
    # x_local: [n_batch, n_lat, n_lon, variables]
    x_patches_local = x.reshape(n_batch, n_x, n_y // self.patch_size, self.patch_size, n_var) # Shape of x_patches: [n_batches, num_patch_x, num_patch_y, size_patch, n_var]
    x_patches_local = x_patches_local.flatten(start_dim=1, end_dim=2)
    x_patches_local = x_patches_local.flatten(start_dim=2, end_dim=3)
    
    q = self.q(x_patches_local) # shape: [n_batches, spatial_tokens, channel_tokens]
    k = self.k(x_patches_local)
    v = self.v(x_patches_local)

    q = self.scale * q
    attention = self.xwt(q, k)
    attention = self.softmax(attention) # TODO: groups for softmax also need to be determined. probably same partners as layernorm
    values = self.xwt(attention, v)    
    return values