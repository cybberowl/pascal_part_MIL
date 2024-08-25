from .conv_block import create_convolution_block
import torch.nn as nn
from .base import get_config_from_locals
import numpy as np

class SegNetEncoder(nn.Module):

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
                    block_depth, enable_batchnorm, act_class, change_channels='first')
            pooling = nn.MaxPool2d(kernel_size=pool_kernel_size,return_indices=True)
            self.conv_blocks.append(conv_block)
            self.poolings.append(pooling)
            in_channels = channels
        
    def forward(self, x):

        pool_idx_array = []
        for conv_block, pooling in zip(self.conv_blocks, self.poolings):
            x = conv_block(x)
            x, idx = pooling(x)
            pool_idx_array.append(idx)

        return x, pool_idx_array

class SegNetDecoder(nn.Module):

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

        for i, channels in enumerate(inv_channels_array):

            conv_block = create_convolution_block(in_channels,channels, conv_kernel_size,
                    block_depth, enable_batchnorm, act_class, change_channels='last')
            unpooling = nn.MaxUnpool2d(kernel_size=pool_kernel_size)
            self.conv_blocks.append(conv_block)
            self.poolings.append(unpooling)
            in_channels = channels

        self.last_layer = nn.Conv2d(in_channels = in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1,
                                    padding = 'same')

    def forward(self, x, inv_pool_idx_array):

        for conv_block, unpooling, idx in zip(self.conv_blocks, self.poolings, inv_pool_idx_array):
            x = unpooling(x, idx)
            x = conv_block(x)

        x = self.last_layer(x)

        return x

class SegNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 encoder_channels_array,
                 decoder_channels_array,
                 pool_kernel_size = 2,
                 conv_kernel_size = 5,
                 block_depth = 3,
                 enable_batchnorm = True,
                 act_class = nn.ReLU):
        super().__init__()
        self.config = get_config_from_locals(**locals())

        assert len(encoder_channels_array) == len(decoder_channels_array)
        assert np.allclose(encoder_channels_array[:-1],decoder_channels_array[:-1][::-1])

        self.Encoder = SegNetEncoder(in_channels,
                                     encoder_channels_array,
                                     pool_kernel_size,
                                     conv_kernel_size,
                                     block_depth,
                                     enable_batchnorm,
                                     act_class)

        self.bottleneck_conv = create_convolution_block(in_channels=encoder_channels_array[-1],
                                                        out_channels=encoder_channels_array[-1],
                                                        kernel_size=conv_kernel_size,
                                                        block_depth = block_depth,
                                                        enable_batchnorm = enable_batchnorm,
                                                        act_class = act_class)


        self.Decoder = SegNetDecoder(encoder_channels_array[-1],
                                     out_channels,
                                     decoder_channels_array,
                                     pool_kernel_size,
                                     conv_kernel_size,
                                     block_depth,
                                     enable_batchnorm,
                                     act_class)

    def forward(self, x):

        e, pool_idx_array = self.Encoder(x)
        inv_pool_idx_array = pool_idx_array[::-1]

        b = self.bottleneck_conv(e)

        d = self.Decoder(b,inv_pool_idx_array)
        return d
