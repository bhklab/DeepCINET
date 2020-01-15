import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1, 100)
            nn.Tanh(),
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50,1),
            nn.ReLU()
        )


    def forward(self, x):
        return self.layer1(x.view(x.size(0), -1))
