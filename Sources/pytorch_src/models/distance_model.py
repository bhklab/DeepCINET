import torch
import torch.nn as nn

class DistanceLayer(nn.Module):
    def __init__(self, hparams):
        super(DistanceLayer, self).__init__()
        self.layers = nn.ModuleList()
        ##WARN: If number of distance layers is larger than 2,
        ##F(x-y) >= 0, F(y-z) >= 0 ==> F(x-z) >=0 does no wold
        for i in range(len(hparams.d_layers) - 1):
            layer = nn.Sequential(
                nn.Linear(hparams.d_layers[i], hparams.d_layers[i+1], bias = False),
                nn.Identity() if i == len(hparams.d_layers) - 2 else nn.Tanh(),
                nn.Dropout(hparams.d_dropout[i])
            )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
