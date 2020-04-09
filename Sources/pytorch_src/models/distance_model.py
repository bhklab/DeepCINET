import torch
import torch.nn as nn

class DistanceLayer(nn.Module):
    def __init__(self):
        super(DistanceLayer, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Linear(32, 16, bias = False),
            nn.Tanh(),
            nn.Linear(16, 16, bias = False),
            nn.Tanh(),
            nn.Linear(16, 8, bias = False),
            nn.Tanh(),
            nn.Linear(8, 1, bias = False)
        )

    def forward(self, x):
        return self.layer1(x)
