import torch
import torch.nn as nn

from pytorch_src.models.conv_model import ConvolutionLayer
from pytorch_src.models.fc_model import FullyConnected

class ImageSiamese(nn.Module):
    def __init__(self):
        super(ImageSiamese, self).__init__()
        self.convolution = ConvolutionLayer()
        self.fc = FullyConnected()

    def forward(self, x, pA, pB):
        x = self.fc(self.convolution(x))
        xA = torch.index_select(x, 0, pA)
        xB = torch.index_select(x, 0, pB)
        z = torch.sub(xA, xB)
        return torch.sigmoid(z)

    def uses_images(self):
        return True
