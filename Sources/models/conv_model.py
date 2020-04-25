import torch
import torch.nn as nn

from pytorch_src.models.residualblock import ResidualBlock3d
from pytorch_src.models.bottleneckblock import BottleneckBlock3d

class ConvolutionLayer(nn.Module):
    def __init__(self, hparams):
        super(ConvolutionLayer, self).__init__()

        # Define the convolutions
        channels_size = hparams.conv_layers
        self.layers = nn.ModuleList()
        for i in range(len(channels_size) - 1):
            layer = nn.Sequential(
                self.ConvModel(channels_size[i], channels_size[i+1], hparams.conv_model),
                nn.MaxPool3d(2)
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def ConvModel(self, in_channels, out_channels, model="Bottleneck"):
        if(model == "Bottleneck"):
            return BottleneckBlock3d(in_channels, 4 * out_channels, out_channels, 3)
        if(model == "ResNet"):
            return ResidualBlock3d(in_channels, out_channels, 3)
        if(model == "Convolution"):
            return nn.Conv3d(in_channels, out_channels, 3, padding = 1)



