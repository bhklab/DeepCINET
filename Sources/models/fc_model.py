import torch
import torch.nn as nn

class FullyConnected(nn.Module):
    def __init__(self, hparams):
        super(FullyConnected, self).__init__()
        layers_size = hparams.fc_layers
        dropout = hparams.dropout
        self.layers = nn.ModuleList();
        for i in range(len(layers_size)-1):
            layer1 = nn.Sequential(
                Residual(layers_size[i], layers_size[i+1], last_layer = (i+1 == len(layers_size) - 1)),
                nn.Dropout(hparams.dropout[i])
            )
            self.layers.append(layer1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            x = layer(x)
        return x

class Residual(nn.Module):
    def __init__(self, in_size, out_size, last_layer = False):
        super(Residual, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size)
        self.fc2 = nn.Linear(out_size, out_size)
        self.last_layer = last_layer
        self.activation = nn.LeakyReLU()
        self.shortcut = nn.Linear(in_size, out_size, bias = False) if in_size != out_size else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        x += residual
        return self.activation(x) if not self.last_layer else self.activation(x)
