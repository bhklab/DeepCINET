import torch
import torch.nn as nn

class ResidualBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super(ResidualBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size = kernel_size, padding = kernel_size//2)
        self.bn1 = nn.BatchNorm3d(out_channels)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, padding = kernel_size//2)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act = nn.LeakyReLU()

        self.shortcut = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size = 1, bias=False),
            nn.BatchNorm3d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.act(x)
        return x
