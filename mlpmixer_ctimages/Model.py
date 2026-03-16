from perun import monitor
import numpy as np
import torch
import torch.nn as nn

from mlpmixer import MLPMixerEncoderLayer, MLPMixerDecoderLayer


class ModelMLPMixer(torch.nn.Module):
    def __init__(self, channels_in, input_resolution = (512,512),
                 classes=2, mlp_ratio=4.0, kernel_flex = False,
                 start_channels = 16):
        super().__init__()

        self.input_resolution = input_resolution
        self.channels_in = channels_in
        self.classes = classes
        self.kernel_flex = kernel_flex
        c = start_channels

        self.conv = nn.Conv2d(in_channels=channels_in,
                              out_channels=c,
                              kernel_size=3,
                              padding=1,
                              stride=2,
                              bias=False)
        
        self.norm = nn.InstanceNorm2d(32)
        self.act = nn.GELU()

        current_res = [input_resolution[0] // 2, input_resolution[1] // 2]

        self.down1 = MLPMixerEncoderLayer(in_channels=c, out_channels=c*2, 
                                         input_resolution=current_res, 
                                         dropout=0.1,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2]

        self.down2 = MLPMixerEncoderLayer(in_channels=c*2, out_channels=c*4, 
                                         input_resolution=current_res, 
                                         dropout=0.1,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2] 

        self.down3 = MLPMixerEncoderLayer(in_channels=c*4, out_channels=c*8, 
                                         input_resolution=current_res, 
                                         dropout=0.2,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex
                                         )
        
        current_res = [current_res[0] // 2, current_res[1] // 2]
       
        self.down4 = MLPMixerEncoderLayer(in_channels=c*8, out_channels=c*16, 
                                         input_resolution=current_res, 
                                         dropout=0.3,
                                         mlp_ratio=mlp_ratio,
                                         kernel_flex=kernel_flex
                                         )
        
        bottleneck_res = [current_res[0] // 2, current_res[1] // 2]

        self.up1 = MLPMixerDecoderLayer(in_channels=c*16, out_channels=c*8, 
                                       input_resolution=bottleneck_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex,
                                       )
        current_res = [bottleneck_res[0] * 2, bottleneck_res[1] * 2]

        self.up2 = MLPMixerDecoderLayer(in_channels=c*8, out_channels=c*4, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex
                                    )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.up3 = MLPMixerDecoderLayer(in_channels=c*4, out_channels=c*2, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex
                                       )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.up4 = MLPMixerDecoderLayer(in_channels=c*2, out_channels=c, 
                                       input_resolution=current_res, 
                                       dropout=0.0,
                                       mlp_ratio=mlp_ratio,
                                       kernel_flex=kernel_flex
                                       )
        current_res = [current_res[0] * 2, current_res[1] * 2]

        self.final_upsample = nn.ConvTranspose2d(
            in_channels=c,
            out_channels=classes,
            kernel_size=3, 
            stride=2, 
            padding=1,
            output_padding=1,
            bias=False
            ) 
    @monitor()   
    def forward(self, x):
        # Initial convolution to reduce spatial dimensions and increase channels
        x_init = self.conv(x) 
        x_init = self.norm(x_init)
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