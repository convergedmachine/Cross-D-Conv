import torch.nn as nn
from CrossDConv import CrossDConv

def replace_conv_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            custom_conv = CrossDConv(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                module.stride[0],
                module.padding[0],
                module.dilation[0],
                module.groups,
                module.bias,
                module.padding_mode
            )
            setattr(model, name, custom_conv)
        else:
            replace_conv_layers(module)

def convert2threed(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            custom_conv = nn.Conv3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size[0],
                module.stride[0],
                module.padding[0],
                module.dilation[0],
                module.groups,
                module.bias,
                module.padding_mode
            )
            setattr(model, name, custom_conv)
        elif isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.BatchNorm3d(module.num_features, module.eps, module.momentum, module.affine , module.track_running_stats))
        elif isinstance(module, nn.MaxPool2d):
            setattr(model, name, nn.MaxPool3d(module.kernel_size, module.stride, module.padding, module.dilation, module.return_indices, module.ceil_mode))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            setattr(model, name, nn.AdaptiveAvgPool3d(module.output_size[0]))
        else:
            convert2threed(module)                        