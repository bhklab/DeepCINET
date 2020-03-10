import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1671, 480),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(480, 240),
            nn.LeakyReLU(),
            nn.Linear(240, 128),
        )


    def forward(self, x):
        return self.layer1(x.view(x.size(0), -1))
