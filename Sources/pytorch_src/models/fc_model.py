import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(432, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
        )


    def forward(self, x):
        return self.layer1(x.view(x.size(0), -1))
