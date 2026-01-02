import torch
import torch.nn as nn
import torch.nn.functional as F
from Mixer import MixerBlockSequential

# x: Input Tensor (B, D, H, W, C)
# output: windows of shape (num_windows*B, window_size, window_size, window_size, C)
def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    num_windowsD = D // window_size
    num_windowsH = H // window_size
    num_windowsW = W // window_size

    # example: (B, 8, 8, 8, C) with window_size=2
    # -> (B, 4, 2, 4, 2, 4, 2, C)
    # for each 8 pixels in D, H, W, split into 4 windows of size 2
    x = x.view(B, 
               num_windowsD, window_size,
               num_windowsH, window_size,
               num_windowsW, window_size,  
               C)
    # rearrange to (B, num_windowsD, num_windowsH, num_windowsW, window_size, window_size, window_size, C)
    # because we want each window to be together
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    
    # merge windows: (B * num_windowsD * num_windowsH * num_windowsW, window_size, window_size, window_size, C)
    # because each window is now a separate example in the batch
    windows = x.view(-1, window_size, window_size, window_size, C)

    # for each window, the coordinates are [i,j,k] are automatically handled by the view and permute operations

    return windows

# window reverse turns windows back to original shape, so the image is without windows    
# input: windows of shape (num_windows*B, window_size, window_size, window_size, C)
def window_reverse(windows, window_size, D, H, W):
    num_windows = (D // window_size) * (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]
    
    x = windows.view(B, 
                     D // window_size, H // window_size, W // window_size, 
                     window_size, window_size, window_size, 
                     C)
    
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, C)
    return x


class SwinMLPBlock(nn.Module):
    # dim: number of input channels
    # input_resolution: (D, H, W)
    # window_size: size of the window
    # shift_size: amount to shift the window
    # mlp_ratio: ratio of hidden dimension to input dimension in MLP
    def __init__(self, dim, input_resolution, window_size=4, shift_size=0, 
                 mlp_ratio=4.0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        
        # TODO: Padding wenn die slice anzahl kleiner als window_size ist
        # Handle the case when the input resolution is smaller than the window size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        else:
            self.shift_size = shift_size

        # Calculate number of pixels in each window
        self.num_pixels_in_window = self.window_size ** 3
        
        self.mixer_block1 = MixerBlockSequential(
            tokens_mlp_dim = self.num_pixels_in_window,
            channels_mlp_dim = dim,
            spatial_hidden_dim_fraction = 1.0,
            channels_hidden_dim_fraction = mlp_ratio,
            dropout = 0.0
        )
        self.mixer_block2 = MixerBlockSequential(
            tokens_mlp_dim = self.num_pixels_in_window,
            channels_mlp_dim = dim,
            spatial_hidden_dim_fraction = 1.0,
            channels_hidden_dim_fraction = mlp_ratio,
            dropout = 0.0
        )

    def forward(self, x):
        B, D, H, W, C = x.shape
        window_part = window_partition(x, self.window_size)
        flat_windows = window_part.view(-1, self.num_pixels_in_window, C)  #(num_windows*B, window_size*window_size*window_size, C)
        mlp_mixer = self.mixer_block1(flat_windows)
        mixer_output = mlp_mixer.view(-1, self.window_size, self.window_size, self.window_size, C)
        window_recover = window_reverse(mixer_output, self.window_size, D, H, W)

        if self.shift_size > 0:
             shifted_x = torch.roll( window_recover, 
                                    shifts=(-self.shift_size, -self.shift_size, -self.shift_size),
                                    dims=(1, 2, 3) )
        else:
             shifted_x = window_recover

        window_part2 = window_partition(shifted_x, self.window_size)
        flat_shifted_windows = window_part2.view(-1, self.num_pixels_in_window, C)  #(num_windows*B, window_size*window_size*window_size, C)
        mlp_mixer2 = self.mixer_block2(flat_shifted_windows)
        mixer_output2 = mlp_mixer2.view(-1, self.window_size, self.window_size, self.window_size, C)
        shifted_x_result = window_reverse(mixer_output2, self.window_size, D, H, W) 

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x_result, 
                shifts=(self.shift_size, self.shift_size, self.shift_size),
                dims=(1, 2, 3)
            )
        else:
            x = shifted_x_result

        return x
    
class SwinMLPEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, window_size = 4, stride = 2):
        super().__init__()

        self.downsample_conv = nn.Conv3d(
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = 3,
                                    padding = 1,
                                    stride = stride,
                                    bias = False
                                    )
        
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()

        input_resolution = [i // 2 for i in input_resolution]
        
        self.swinMlpBlock = SwinMLPBlock(out_channels, input_resolution, window_size)

        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x):

        x = self.downsample_conv(x)
        x = self.norm1(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.swinMlpBlock(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.norm2(x)
        x = self.act(x)
    
        return x
     
class SwinMLPDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution, window_size = 4, stride = 2):
        super().__init__()

        self.upsample_conv = nn.ConvTranspose3d(
                                    in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = 2,
                                    stride = stride,
                                    bias = False
                                    )
        
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.act = nn.GELU()

        self.conv = nn.Conv3d(
            in_channels = 2 * out_channels,
            out_channels = out_channels,
            kernel_size = 1,
            bias = False
        )

        input_resolution = [i * 2 for i in input_resolution]
        
        self.swinMlpBlock = SwinMLPBlock(out_channels, input_resolution, window_size)

        self.norm2 = nn.InstanceNorm3d(out_channels)

    def forward(self, x, x_skip):
        
        x = self.upsample_conv(x)
        x = self.norm1(x)
        x = self.act(x)

        # concatenate skip connection
        x = torch.cat((x, x_skip), dim=1)
        x = self.conv(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.swinMlpBlock(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.norm2(x)
        x = self.act(x)
    
        return x
