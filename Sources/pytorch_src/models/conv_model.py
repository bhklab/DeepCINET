import torch
import torch.nn as nn

from pytorch_src.models.residualblock import ResidualBlock3d
from pytorch_src.models.bottleneckblock import BottleneckBlock3d

class ConvolutionLayer(nn.Module):
    def __init__(self):
        super(ConvolutionLayer, self).__init__()

        # Define the convolutions
        self.layer1 = nn.Sequential(
            BottleneckBlock3d(1, 4, 4, 3),
            nn.MaxPool3d(2),
            BottleneckBlock3d(4, 8, 8, 3),
            nn.MaxPool3d(2),
            BottleneckBlock3d(8, 16, 16, 3),
            nn.MaxPool3d(2),
            BottleneckBlock3d(16, 32, 16, 3),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.layer1(x)
