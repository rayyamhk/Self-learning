import torch.nn as nn
from .utils.BottleneckResidualBlock import BottleneckResidualBlock

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn_64 = nn.BatchNorm2d(64)
        self.res2_1 = BottleneckResidualBlock(64, 256)
        self.res2_2 = BottleneckResidualBlock(256, 256)
        self.res2_3 = BottleneckResidualBlock(256, 256)
        self.res3_0 = BottleneckResidualBlock(256, 512, downsampling=True)
        self.res3_1 = BottleneckResidualBlock(512, 512)
        self.res3_2 = BottleneckResidualBlock(512, 512)
        self.res3_3 = BottleneckResidualBlock(512, 512)
        self.res4_0 = BottleneckResidualBlock(512, 1024, downsampling=True)
        self.res4_1 = BottleneckResidualBlock(1024, 1024)
        self.res4_2 = BottleneckResidualBlock(1024, 1024)
        self.res4_3 = BottleneckResidualBlock(1024, 1024)
        self.res4_4 = BottleneckResidualBlock(1024, 1024)
        self.res4_5 = BottleneckResidualBlock(1024, 1024)
        self.res5_0 = BottleneckResidualBlock(1024, 2048, downsampling=True)
        self.res5_1 = BottleneckResidualBlock(2048, 2048)
        self.res5_2 = BottleneckResidualBlock(2048, 2048)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, imgs):
        bs = imgs.shape[0]

        imgs = self.bn_64(self.conv1(imgs))
        imgs = self.relu(imgs)
        
        imgs = self.maxpool(imgs)
        imgs = self.res2_1(imgs)
        imgs = self.res2_2(imgs)
        imgs = self.res2_3(imgs)

        imgs = self.res3_0(imgs)
        imgs = self.res3_1(imgs)
        imgs = self.res3_2(imgs)
        imgs = self.res3_3(imgs)

        imgs = self.res4_0(imgs)
        imgs = self.res4_1(imgs)
        imgs = self.res4_2(imgs)
        imgs = self.res4_3(imgs)
        imgs = self.res4_4(imgs)
        imgs = self.res4_5(imgs)

        imgs = self.res5_0(imgs)
        imgs = self.res5_1(imgs)
        imgs = self.res5_2(imgs)

        imgs = self.avgpool(imgs)
        imgs = self.flatten(imgs)
        imgs = self.fc(imgs)

        return imgs
