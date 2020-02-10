import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(50, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
        )


    def forward(self, x):
        return self.layer1(x.view(x.size(0), -1))
