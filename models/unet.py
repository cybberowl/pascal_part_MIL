import torch.nn as nn
import torch
from .base import get_config_from_locals
from .conv_block import create_convolution_block

class UNetEncoder(nn.Module):

    def __init__(self,
                in_channels,
                channels_array,
                pool_kernel_size,
                conv_kernel_size,
                block_depth,
                enable_batchnorm,
                act_class):

        super().__init__()

        self.conv_blocks = nn.ModuleList()
        self.poolings = nn.ModuleList()

        for i, channels in enumerate(channels_array):
            conv_block = create_convolution_block(in_channels,channels, pool_kernel_size,
                    block_depth, enable_batchnorm, act_class)
            pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            self.conv_blocks.append(conv_block)
            self.poolings.append(pooling)
            in_channels = channels

    def forward(self, x):
        skipcon_features = []
        for conv_block, pooling in zip(self.conv_blocks, self.poolings):
            x = conv_block(x)
            skipcon_features.append(x)
            x = pooling(x)

        return x, skipcon_features

class UNetDecoder(nn.Module):

    def __init__(self,
                in_channels,
                out_channels,
                inv_channels_array,
                pool_kernel_size,
                conv_kernel_size,
                block_depth,
                enable_batchnorm,
                act_class):

        super().__init__()
        self.conv_blocks = nn.ModuleList()
        self.poolings = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        for i, channels in enumerate(inv_channels_array):
            
            upconv = nn.Conv2d(in_channels =  in_channels, out_channels = channels, 
                               kernel_size=conv_kernel_size, padding = 'same')
            conv_block = create_convolution_block(2*channels,channels, conv_kernel_size,
                    block_depth, enable_batchnorm, act_class)
            unpooling = nn.Upsample(scale_factor=pool_kernel_size)
            
            self.upconvs.append(upconv)
            self.conv_blocks.append(conv_block)
            self.poolings.append(unpooling)
            in_channels = channels
            
        self.last_layer = nn.Conv2d(in_channels = in_channels,
                                    out_channels=out_channels, 
                                    kernel_size=1,
                                    padding = 'same')

    def forward(self, x, skipcon_features):

        for conv_block, unpooling,upconv, skf in zip(
                self.conv_blocks, self.poolings, self.upconvs, skipcon_features):
            x = unpooling(x)
            x = upconv(x)
            x = torch.concat([skf,x],dim = 1)
            x = conv_block(x)
            
        x = self.last_layer(x)

        return x

class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 channels_array,
                 bottleneck_channels,
                 pool_kernel_size = 2,
                 conv_kernel_size = 3,
                 block_depth = 2,
                 enable_batchnorm = True,
                 act_class = nn.ReLU):
        super().__init__()
        self.config = get_config_from_locals(**locals())

        self.Encoder = UNetEncoder(in_channels,
                                     channels_array,
                                     pool_kernel_size,
                                     conv_kernel_size,
                                     block_depth,
                                     enable_batchnorm,
                                     act_class)

        self.bottleneck_conv = create_convolution_block(in_channels=channels_array[-1],
                                                        out_channels=bottleneck_channels,
                                                        kernel_size=conv_kernel_size,
                                                        block_depth = block_depth,
                                                        enable_batchnorm = enable_batchnorm,
                                                        act_class = act_class)


        self.Decoder = UNetDecoder(bottleneck_channels,
                                   out_channels,
                                     channels_array[::-1],
                                     pool_kernel_size,
                                     conv_kernel_size,
                                     block_depth,
                                     enable_batchnorm,
                                     act_class)

    def forward(self, x):
        # encoder
        e, skipcon_features = self.Encoder(x)
        inv_skipcon_features = skipcon_features[::-1]

        # bottleneck
        b = self.bottleneck_conv(e)

        # decoder
        d = self.Decoder(b,inv_skipcon_features)
        return d