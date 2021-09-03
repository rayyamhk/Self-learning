import torch.nn as nn

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsampling=False):
        super().__init__()

        if downsampling == False:
            stride = 1
            bottleneck = 1 if in_channels == 64 else 4
        else:
            stride = 2
            bottleneck = 2

        self.skip_connection = nn.Sequential()

        self.conv_1 = nn.Conv2d(in_channels, in_channels // bottleneck, 1)
        self.conv_2 = nn.Conv2d(in_channels // bottleneck, in_channels // bottleneck, 3, stride=stride, padding=1)
        self.conv_3 = nn.Conv2d(in_channels // bottleneck, out_channels, 1)
        self.bn_1 = nn.BatchNorm2d(in_channels // bottleneck)
        self.bn_2 = nn.BatchNorm2d(in_channels // bottleneck)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.skip_connection.add_module('conv_0', nn.Conv2d(in_channels, out_channels, 1, stride=stride))
            self.skip_connection.add_module('bn_0', nn.BatchNorm2d(out_channels))

    def forward(self, imgs):
        sc = self.skip_connection(imgs)

        imgs = self.bn_1(self.conv_1(imgs))
        imgs = self.relu(imgs)

        imgs = self.bn_2(self.conv_2(imgs))
        imgs = self.relu(imgs)

        imgs = self.bn_3(self.conv_3(imgs))
        imgs += sc
        imgs = self.relu(imgs)

        return imgs

