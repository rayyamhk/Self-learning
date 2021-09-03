import torch.nn as nn
from .utils.ResidualBlock import ResidualBlock

"""
Source: https://arxiv.org/pdf/1512.03385.pdf
ResNet20 architecture is designed for analyse the performance of ResNet with different number of layers on CIFAR-10 datasets
Input size should be small (32 x 32)
"""

class ResNet20(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn_16 = nn.BatchNorm2d(16)
        self.res2_1 = ResidualBlock(16, 16)
        self.res2_2 = ResidualBlock(16, 16)
        self.res2_3 = ResidualBlock(16, 16)
        self.res3_0 = ResidualBlock(16, 32, downsampling=True)
        self.res3_1 = ResidualBlock(32, 32)
        self.res3_2 = ResidualBlock(32, 32)
        self.res4_0 = ResidualBlock(32, 64, downsampling=True)
        self.res4_1 = ResidualBlock(64, 64)
        self.res4_2 = ResidualBlock(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, num_classes)

    def forward(self, imgs):
        bs = imgs.shape[0]
        assert imgs.shape == (bs, 3, 32, 32), "Incorrect shape for input image"

        imgs = self.bn_16(self.conv1(imgs))
        imgs = self.relu(imgs)
        assert imgs.shape == (bs, 16, 32, 32), "Incorrect shape for output of layer 1"
        
        imgs = self.res2_1(imgs)
        imgs = self.res2_2(imgs)
        imgs = self.res2_3(imgs)
        assert imgs.shape == (bs, 16, 32, 32), "Incorrect shape for output of layer 2"

        imgs = self.res3_0(imgs)
        imgs = self.res3_1(imgs)
        imgs = self.res3_2(imgs)
        assert imgs.shape == (bs, 32, 16, 16), "Incorrect shape for output of layer 3"

        imgs = self.res4_0(imgs)
        imgs = self.res4_1(imgs)
        imgs = self.res4_2(imgs)
        assert imgs.shape == (bs, 64, 8, 8), "Incorrect shape for output of layer 4"

        imgs = self.avgpool(imgs)
        imgs = self.flatten(imgs)
        imgs = self.fc(imgs)
        assert imgs.shape == (bs, 10), "Incorrect shape for output of classification layer"

        return imgs
