import torch.nn as nn


class Aerts(nn.Module):
    def __init__(self, height, width, depth):
        super(Aerts, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=depth, out_channels=64, kernel_size=5)
        self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool1 = nn.MaxPool3d(kernel_size=3)
        self.conv3 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv3d(in_channels=256, out_channels=512, kernel_size=3)
        self.pool2 = nn.MaxPool3d(kernel_size=3)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(13824, 13824)
        self.fc2 = nn.Linear(13824, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.BatchNorm2d(self.conv1.out_channels)(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(self.conv2.out_channels)(x)
        x = self.leaky_relu(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = nn.BatchNorm2d(self.conv3.out_channels)(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = nn.BatchNorm2d(self.conv4.out_channels)(x)
        x = self.leaky_relu(x)
        x = self.pool2(x)

        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = nn.BatchNorm1d(self.fc1.out_features)(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = nn.BatchNorm1d(self.fc2.out_features)(x)
        x = self.leaky_relu(x)
        x = self.fc3(x)
        x = nn.BatchNorm1d(self.fc3.out_features)(x)
        x = self.leaky_relu(x)
        x = self.fc4(x)

        return x

