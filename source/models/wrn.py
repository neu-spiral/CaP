import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

import sys
import numpy as np

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['wrn']

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate, conv_layer, bn_layer, bn_partition):
        super(BasicBlock, self).__init__()
        self.bn1 = bn_layer(in_planes) if bn_partition==1 else bn_layer(in_planes,bn_partition)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn_layer(out_planes) if bn_partition==1 else bn_layer(out_planes,bn_partition)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate, conv_layer, bn_layer, bn_partition):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, conv_layer, bn_layer, bn_partition)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, conv_layer, bn_layer, bn_partition):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, conv_layer, bn_layer, bn_partition))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropRate, conv_layer, bn_layer, num_classes, bn_partition=[1]*9, shrink=1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, conv_layer, bn_layer, bn_partition.pop(0))
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, conv_layer, bn_layer, bn_partition.pop(0))
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, conv_layer, bn_layer, bn_partition.pop(0))
        # global average pooling and classifier
        num_bn = bn_partition.pop(0)
        self.bn1 = bn_layer(nChannels[3]) if num_bn==1 else bn_layer(nChannels[3], num_bn)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

def wrn16_8(conv_layer, bn_layer, **kwargs):
    bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*9
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(16, 8, 0.0, conv_layer, bn_layer, num_classes=kwargs['num_classes'], bn_partition=bn_partition, shrink=shrink)

def wrn28_10(conv_layer, bn_layer, **kwargs):
    bn_partition = kwargs['bn_partition'] if 'bn_partition' in kwargs else [1]*13
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(28, 10, 0.3, conv_layer, bn_layer, kwargs['num_classes'], bn_partition=bn_partition, shrink=shrink)

def wrn28_4(conv_layer, bn_layer, **kwargs):
    rob = kwargs['robustness'] if 'robustness' in kwargs else False
    shrink = kwargs['shrink'] if 'shrink' in kwargs else 1
    return WideResNet(28, 4, 0, kwargs['num_classes'], rob=rob, shrink=shrink)

if __name__ == '__main__':
    net=WideResNet(28, 10, 0, 100, True, 1)
    #y = net(Variable(torch.randn(1,3,32,32)))
    #print(y.size())
    summary(net.cuda(), (3,32,32))