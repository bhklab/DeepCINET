import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageSiamese(nn.Module):
    def __init__(self):
        self.convolution = ConvolutionLayer()
        self.fc = FullyConnected()

    def forward(self, x, y):
        x = self.fc(self.convolution(x))
        y = self.fc(self.convolution(y))
        z = torch.sub(x, y)
        return F.sigmoid(z)

    def uses_images():
        return true
