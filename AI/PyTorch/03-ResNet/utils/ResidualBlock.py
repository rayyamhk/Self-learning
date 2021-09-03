import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=False):
        super().__init__()

        stride = 1 if downsampling == False else 2

        self.skip_connection = nn.Sequential()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.skip_connection.add_module('conv_0', nn.Conv2d(in_channels, out_channels, 1, stride=2))
            self.skip_connection.add_module('bn_0', nn.BatchNorm2d(out_channels))

    def forward(self, imgs):
        sc = self.skip_connection(imgs)

        imgs = self.bn_1(self.conv_1(imgs))
        imgs = self.relu(imgs)

        imgs = self.bn_2(self.conv_2(imgs))
        imgs += sc
        imgs = self.relu(imgs)

        return imgs

