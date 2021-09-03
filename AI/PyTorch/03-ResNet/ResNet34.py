import torch.nn as nn
from .utils.ResidualBlock import ResidualBlock

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn_64 = nn.BatchNorm2d(64)
        self.res2_1 = ResidualBlock(64, 64)
        self.res2_2 = ResidualBlock(64, 64)
        self.res2_3 = ResidualBlock(64, 64)
        self.res3_0 = ResidualBlock(64, 128, downsampling=True)
        self.res3_1 = ResidualBlock(128, 128)
        self.res3_2 = ResidualBlock(128, 128)
        self.res3_3 = ResidualBlock(128, 128)
        self.res4_0 = ResidualBlock(128, 256, downsampling=True)
        self.res4_1 = ResidualBlock(256, 256)
        self.res4_2 = ResidualBlock(256, 256)
        self.res4_3 = ResidualBlock(256, 256)
        self.res4_4 = ResidualBlock(256, 256)
        self.res4_5 = ResidualBlock(256, 256)
        self.res5_0 = ResidualBlock(256, 512, downsampling=True)
        self.res5_1 = ResidualBlock(512, 512)
        self.res5_2 = ResidualBlock(512, 512)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, num_classes)

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
