import torch
import torch.nn as nn

class ConvolutionLayer(nn.Module):
    def __init__(self):
        super(ConvolutionLayer, self).__init__()

        # Define the convolutions
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, 20, kernel_size = 3, stride = 2, padding = 0),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(20, 20, kernel_size = 2, stride = 2, padding = 0),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.MaxPool3d(3)
            )

        self.layer2 = nn.Sequential(
            nn.Conv3d(20, 40, kernel_size = 3, stride = 1, padding = 0),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Conv3d(40, 40, kernel_size = 2, stride = 1, padding = 0),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))
