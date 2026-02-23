import torch
import torch.nn as nn
import torch.nn.functional as F
from Mixer import MixerBlockSequential

# x: Input Tensor (B, H, W, C)
# output: windows of shape (num_windows*B, window_size, window_size, window_size, C)
def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0, f"Height {H} not divisible by {window_size}"
    assert W % window_size == 0, f"Width {W} not divisible by {window_size}"

    # example: (B, 8, 8, C) with window_size=2
    # -> (B, 4, 2, 4, 2, C)
    # for each 8 pixels in H, W, split into 4 windows of size 2
    x = x.view(B,
                H // window_size, window_size,
                W // window_size, window_size, 
                C)
    # rearrange to (B, num_windowsH, num_windowsW, window_size, window_size, window_size, C)
    # because we want each window to be together
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    
    # merge windows: (B * num_windowsH * num_windowsW, window_size, window_size, C)
    # because each window is now a separate example in the batch
    windows = x.view(-1, window_size, window_size, C)

    return windows

# window reverse turns windows back to original shape, so the image is without windows    
# input: windows of shape (num_windows*B, window_size, window_size, C)
def window_reverse(windows, window_size, H, W):
    num_windows =  (H // window_size) * (W // window_size)
    B = windows.shape[0] // num_windows
    C = windows.shape[-1]
    
    x = windows.view(B, 
                     H // window_size, W // window_size, 
                     window_size, window_size, 
                     C)
    
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, C)
    return x


class SwinMLPBlock(nn.Module):
    # dim: number of input channels
    # input_resolution:  (H, W)
    # window_size: size of the window
    # shift_size: amount to shift the window
    # mlp_ratio: ratio of hidden dimension to input dimension in MLP
    def __init__(self, dim, input_resolution, window_size=4, shift_size=0, 
                 mlp_ratio=4.0, dropout=0.0, shifted_windows=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shifted_windows = shifted_windows
        
        if shifted_windows:
            if shift_size == 0:
                self.shift_size = window_size // 2
            else:
                self.shift_size = shift_size
        else:
            self.shift_size = 0
        
        # Handle the case when the input resolution is smaller than the window size
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # Calculate number of pixels in each window
        self.num_pixels_in_window = self.window_size ** 2
        
        self.mixer_block1 = MixerBlockSequential(
            tokens_mlp_dim = self.num_pixels_in_window,
            channels_mlp_dim = dim,
            spatial_hidden_dim_fraction = 1.0,
            channels_hidden_dim_fraction = mlp_ratio,
            dropout = dropout
        )
        self.mixer_block2 = MixerBlockSequential(
            tokens_mlp_dim = self.num_pixels_in_window,
            channels_mlp_dim = dim,
            spatial_hidden_dim_fraction = 1.0,
            channels_hidden_dim_fraction = mlp_ratio,
            dropout = dropout
        )

    def forward(self, x):
        B, H, W, C = x.shape
        window_part = window_partition(x, self.window_size)
        flat_windows = window_part.view(-1, self.num_pixels_in_window, C)  #(num_windows*B, window_size*window_size, C)
        mlp_mixer = self.mixer_block1(flat_windows)
        mixer_output = mlp_mixer.view(-1, self.window_size, self.window_size, C)
        window_recover = window_reverse(mixer_output, self.window_size, H, W)

        if self.shift_size > 0:
             shifted_x = torch.roll( window_recover, 
                                    shifts=(-self.shift_size, -self.shift_size),
                                    dims=(1, 2) )
        else:
             shifted_x = window_recover

        window_part2 = window_partition(shifted_x, self.window_size)
        flat_shifted_windows = window_part2.view(-1, self.num_pixels_in_window, C)  #(num_windows*B, window_size*window_size*window_size, C)
        mlp_mixer2 = self.mixer_block2(flat_shifted_windows)
        mixer_output2 = mlp_mixer2.view(-1, self.window_size, self.window_size, C)
        shifted_x_result = window_reverse(mixer_output2, self.window_size, H, W) 

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x_result, 
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2)
            )
        else:
            x = shifted_x_result

        return x
    
class SwinMLPEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution,
                window_size = 4, dropout=0.0, mlp_ratio=4.0,
                shifted_windows=False, kernel_flex=False):
        super().__init__()

        self.pool = nn.MaxPool2d(stride=2, kernel_size=2)

        if kernel_flex:
            current_res = input_resolution[0]
            kernel_size = int(3 * (current_res / 256))

            if kernel_size % 2 == 0: kernel_size += 1 
            if kernel_size < 3: kernel_size = 3

            padding = kernel_size // 2
        else:
            kernel_size = 3
            padding = 1

        self.conv = nn.Conv2d(in_channels = in_channels,
                            out_channels = out_channels,
                            kernel_size = kernel_size,
                            padding = padding,
                            stride = 1,
                            bias = False
                            )

        self.act = nn.GELU()

        self.current_resolution = [i // 2 for i in input_resolution]
        
        self.swinMlpBlock = SwinMLPBlock(dim=out_channels, input_resolution=self.current_resolution,
                                        window_size=window_size, dropout=dropout,
                                        mlp_ratio=mlp_ratio, shifted_windows=shifted_windows)


    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1)
        x = self.swinMlpBlock(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        return x
     
class SwinMLPDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution,
                window_size = 4, dropout=0.0, mlp_ratio=4.0,
                shifted_windows=False, kernel_flex=False, skip = True):
        super().__init__()
        self.skip = skip

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        conv_in = out_channels * 2 if skip else out_channels

        if kernel_flex:
            current_res = input_resolution[0] * 2       # because of upsample, so resolution doubles
            kernel_size = int(3 * (current_res / 256))

            if kernel_size % 2 == 0: kernel_size += 1 
            if kernel_size < 3: kernel_size = 3

            padding = kernel_size // 2
        else:
            kernel_size = 3
            padding = 1

        self.conv = nn.Conv2d(
            in_channels=conv_in,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
            
        self.act = nn.GELU()

        self.current_resolution = [i * 2 for i in input_resolution]
        
        self.swinMlpBlock = SwinMLPBlock(dim=out_channels,
                                         input_resolution=self.current_resolution, 
                                         window_size=window_size, 
                                         dropout=dropout, 
                                         mlp_ratio=mlp_ratio,
                                         shifted_windows=shifted_windows
                                         )

    def forward(self, x, x_skip = None):
        x1 = self.up(x)

        if self.skip and x_skip is not None:
            diffY = x_skip.size()[2] - x1.size()[2]
            diffX = x_skip.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                             diffY // 2, diffY - diffY // 2])
            x1 = torch.cat([x1, x_skip], dim=1) 

        x = self.conv(x1)
        x = self.act(x)

        x = x.permute(0, 2, 3, 1)
        x = self.swinMlpBlock(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        return x
