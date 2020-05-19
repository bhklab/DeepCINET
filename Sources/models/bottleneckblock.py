import torch
import torch.nn as nn

from models.squeezeandexcite import SqueezeAndExcite

class BottleneckBlock3d(nn.Module):
    def __init__(self, in_channels, expansion_channels, out_channels, kernel_size):
        super(BottleneckBlock3d, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv3d(in_channels, expansion_channels, kernel_size = 1),
            nn.BatchNorm3d(expansion_channels),
            nn.LeakyReLU(inplace = True)
        )

        self.dwconv = nn.Sequential(
            nn.Conv3d(expansion_channels, expansion_channels, kernel_size = kernel_size, padding=1, groups = expansion_channels),
            nn.BatchNorm3d(expansion_channels),
            nn.LeakyReLU(inplace = True)
        )

        self.convsqueeze = nn.Sequential(
            nn.Conv3d(expansion_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels)
        )

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 1, bias=False),
            nn.BatchNorm3d(out_channels)) if in_channels != out_channels else nn.Identity()
        self.SE = SqueezeAndExcite(out_channels)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.SE(self.convsqueeze(self.dwconv(self.conv1x1(x))))
        x += residual
        return x
