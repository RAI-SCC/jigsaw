import torch
import torch.nn as nn
import torch.nn.functional as F
from Jigsaw.Mixer import MixerBlockSequential
    
class MLPMixerBlock(nn.Module):
    def __init__(self, dim, input_resolution, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.H, self.W = input_resolution
        self.num_tokens = self.H * self.W

        self.mixer_block = MixerBlockSequential(
            tokens_mlp_dim=self.num_tokens,
            channels_mlp_dim=dim,
            spatial_hidden_dim_fraction=1.0,
            channels_hidden_dim_fraction=mlp_ratio,
            dropout=dropout
        )

    def forward(self, x):
        B, H, W, C = x.shape
        # Flatten spatial dims → (B, H*W, C)
        x_flat = x.view(B, H * W, C)
        # MLP-Mixer auf allen Tokens
        x_mixed = self.mixer_block(x_flat)
        # Zurück zu (B, H, W, C)
        x_out = x_mixed.view(B, H, W, C)
        return x_out
    
class MLPMixerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution,
                 dropout=0.0, mlp_ratio=4.0, kernel_flex=False):
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
        
        self.MlpBlock = MLPMixerBlock(dim=out_channels, input_resolution=self.current_resolution,
                                      dropout=dropout, mlp_ratio=mlp_ratio)


    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.MlpBlock(x)
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = self.act(x)
        return x
     
class MLPMixerDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution,
                dropout=0.0, mlp_ratio=4.0,
                kernel_flex=False, skip = True):
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
        
        self.MlpBlock = MLPMixerBlock(dim=out_channels,
                                         input_resolution=self.current_resolution, 
                                         dropout=dropout, 
                                         mlp_ratio=mlp_ratio
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
        x = self.MlpBlock(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        return x
