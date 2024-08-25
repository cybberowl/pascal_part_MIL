import torch.nn as nn

def create_convolution_block(in_channels, out_channels, kernel_size, block_depth,
        enable_batchnorm, act_class, change_channels = 'first'):

    res = nn.Sequential()
    cin = in_channels
    cout = in_channels


    # first for Segnet encoder and last for Segnet decoder
    # first for UNet for both encoder and decoder
    assert(change_channels in ['first','last']) 

    index_change_channels = 0 if change_channels == 'first' else block_depth-1

    for i in range(block_depth):
        if i == index_change_channels:
            cout = out_channels
        layer = nn.Sequential()
        layer.add_module(f'conv', nn.Conv2d(in_channels = cin,out_channels = cout,
                            kernel_size = kernel_size, padding = 'same'))
        cin = cout

        if enable_batchnorm:
            layer.add_module(f'batchnorm',nn.BatchNorm2d(num_features=cout))

        layer.add_module(f'activation', act_class())
        res.add_module(f'conv_layer_{i}', layer)

    return res