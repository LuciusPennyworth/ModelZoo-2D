"""
Created on 2020/6/15 11:27 周一
@author: Matt zhuhan1401@126.com
Description: description
"""

import torch
import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = self.downsample(x) if self.downsample else x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=10,):
        super(ResNet, self).__init__()    
        self.inplanes = 128
        # input size 32*32          stride=1 so the output size will be same as input
        self.conv1 = conv3x3(3, 64, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = conv3x3(64, 64, stride=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = conv3x3(64, 128, stride=1)
        self.bn3 = nn.BatchNorm2d(128)
        # 32*32

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 4*4
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inchannel, num_block, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != inchannel * block.expansion :
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, inchannel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(inchannel * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, inchannel, stride, downsample))
        self.inplanes = inchannel * block.expansion
        for i in range(1, num_block):
            layers.append(block(self.inplanes, inchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


def resnet18(classes):
    model = ResNet(BasicBlock, [2, 2, 2, 2,], classes)
    return model












