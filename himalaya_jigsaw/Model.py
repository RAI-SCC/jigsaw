from perun import monitor
import numpy as np
import torch
import torch.nn as nn

from distributed_mlpmixer import MLPMixerEncoderLayer, MLPMixerDecoderLayer
from Jigsaw.Convolutions import DistributedConv2D, DistributedConvTranspose2D
from Jigsaw.Normalization import DistributedLayerNorm


class ModelMLPMixer(torch.nn.Module):
    def __init__(self, channels_in, input_resolution = (512,512),
                 classes=2, mlp_ratio=4.0, kernel_flex = False, rank=1,
                 device='cuda', n_parallel=1, model_group=None,
                 spatial_group=None, channel_group=None):
        super().__init__()

        self.input_resolution = input_resolution
        self.channels_in = channels_in
        self.classes = classes
        self.kernel_flex = kernel_flex

        self.conv = DistributedConv2D(in_channels=channels_in,
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    dilation=1,
                                    groups=1,
                                    rank=rank,
                                    spatial_group=spatial_group,
                                    channel_group=channel_group,
                                    device=device,
                                    parallelism=n_parallel,
                                    bias=False
                                )
        
        self.norm = DistributedLayerNorm(dim = 32, rank=rank, device=device, model_group=model_group)
        self.act = nn.GELU()

        current_res = [input_resolution[0] // 2, input_resolution[1] // 2]

        self.down1 = MLPMixerEncoderLayer(in_channels=32, out_channels=64, 
                                         input_resolution=current_res, 
                                         dropout=0.1,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex,
                                         rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                            spatial_group=spatial_group, channel_group=channel_group
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2]

        self.down2 = MLPMixerEncoderLayer(in_channels=64, out_channels=128, 
                                         input_resolution=current_res, 
                                         dropout=0.1,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex,
                                         rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                         spatial_group=spatial_group, channel_group=channel_group
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2] 

        self.down3 = MLPMixerEncoderLayer(in_channels=128, out_channels=256, 
                                         input_resolution=current_res, 
                                         dropout=0.2,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex,
                                         rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                         spatial_group=spatial_group, channel_group=channel_group
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2]
       
        self.down4 = MLPMixerEncoderLayer(in_channels=256, out_channels=512, 
                                         input_resolution=current_res, 
                                         dropout=0.3,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex,
                                         rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                         spatial_group=spatial_group, channel_group=channel_group
                                         )
        
        bottleneck_res = [current_res[0] // 2, current_res[1] // 2]

        self.up1 = MLPMixerDecoderLayer(in_channels=512, out_channels=256, 
                                       input_resolution=bottleneck_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex,
                                       rank=rank, device=device,
                                       n_parallel=n_parallel, model_group=model_group,
                                       spatial_group=spatial_group, channel_group=channel_group
                                       )
        current_res = [bottleneck_res[0] * 2, bottleneck_res[1] * 2]

        self.up2 = MLPMixerDecoderLayer(in_channels=256, out_channels=128, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex,
                                       rank=rank, device=device,
                                       n_parallel=n_parallel, model_group=model_group,
                                       spatial_group=spatial_group, channel_group=channel_group
                                    )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.up3 = MLPMixerDecoderLayer(in_channels=128, out_channels=64, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex,
                                       rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                         spatial_group=spatial_group, channel_group=channel_group
                                       )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.up4 = MLPMixerDecoderLayer(in_channels=64, out_channels=32, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex,
                                       rank=rank, device=device,
                                         n_parallel=n_parallel, model_group=model_group,
                                         spatial_group=spatial_group, channel_group=channel_group
                                       )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.final_upsample = DistributedConvTranspose2D(
            in_channels=32,
            out_channels=classes,
            kernel_size=3, 
            stride=2, 
            padding=1,
            dilation=1,
            groups=1,
            output_padding=1,
            bias=False,
            rank=rank, device=device,
            spatial_group=spatial_group,
            channel_group=channel_group,
            parallelism= n_parallel
            ) 
        
    @monitor()   
    def forward(self, x):
        # Initial convolution to reduce spatial dimensions and increase channels
        x_init = self.conv(x)
        x_init = x_init.permute(0, 2, 3, 1)  # (B,C,H,W) → (B,H,W,C)
        x_init = self.norm(x_init)
        x_init = x_init.permute(0, 3, 1, 2)  # (B,H,W,C) → (B,C,H,W)
        x_init = self.act(x_init)

        # Encoding path
        x1 = self.down1(x_init)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x_bottleneck = self.down4(x3)

        # Decoding path with skip connections
        x = self.up1(x_bottleneck, x_skip=x3)
        x = self.up2(x, x_skip=x2)
        x = self.up3(x, x_skip=x1)
        x = self.up4(x, x_skip=x_init)

        logits = self.final_upsample(x)
        return logits