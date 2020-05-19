import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, r = 4):
        super(SqueezeAndExcite, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // r)
        self.fc2 = nn.Linear(in_channels // r, in_channels)
        self.avgPool = nn.AdaptiveAvgPool3d(1)
        self.act = nn.ReLU(inplace = True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        scale = self.avgPool(x).view(x.size(0), x.size(1))
        scale = self.act(self.fc1(scale))
        scale = self.sig(self.fc2(scale))
        scale = scale.view(x.size(0), x.size(1), 1, 1, 1)
        return x * scale.expand_as(x)
