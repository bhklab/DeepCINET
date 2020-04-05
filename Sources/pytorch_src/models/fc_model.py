import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1556, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
        )


    def forward(self, x):
        return self.layer1(x.view(x.size(0), -1))
