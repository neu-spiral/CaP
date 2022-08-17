'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, pool_kernel=2, pool_padding=0, shrink=1):
        super(BasicBlock, self).__init__()
        self.shrink = shrink
        self.conv1 = nn.Conv2d(int(self.shrink*in_planes), planes, kernel_size=(3,3), padding="same")
        self.conv2 = nn.Conv2d(int(self.shrink*planes), planes, kernel_size=(3,3), padding="same")
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.pool = nn.MaxPool2d(pool_kernel, padding=pool_padding)
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = out[:,:int(self.shrink*out.size(1)),:,:] 
        out = self.conv2(out)
        out = out[:,:int(self.shrink*out.size(1)),:,:]
        out += self.shortcut(x)
        out = self.pool(F.relu(out))
        
        return out

class ResLike(nn.Module):
    def __init__(self, block, num_classes=10, mode=''):
        super(ResLike, self).__init__()
        self.mode = mode
        self.shrink = 0.2 if mode == 'calcomp' else 1
        
        self.conv1 = nn.Conv2d(int(self.shrink*15), 32, kernel_size=(7,7), padding="same")
        self.conv2 = nn.Conv2d(int(self.shrink*32), 32, kernel_size=(3,3), padding="same")
        self.layer1 = nn.Sequential(block(32, 32, pool_kernel=2, shrink=self.shrink))
        self.layer2 = nn.Sequential(block(32, 32, pool_kernel=2, shrink=self.shrink))
        self.layer3 = nn.Sequential(block(32, 32, pool_kernel=2, shrink=self.shrink))
        self.layer4 = nn.Sequential(block(32, 32, pool_kernel=2, shrink=self.shrink))
        
        #self.hidden1 = nn.Linear(4160, 512)
        self.hidden1 = nn.Linear(int(self.shrink*10240), int(self.shrink*512))
        self.hidden2 = nn.Linear(int(self.shrink*512), int(self.shrink*256))
        self.out = nn.Linear(int(self.shrink*256), num_classes)  # 128
        #######################
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.pool2 = nn.AvgPool2d(2)
    
        
        
    def forward(self, *x):
        if self.mode == 'calcomp':
            x = x[4]
        elif isinstance(x, tuple) and len(x)>1:
            x = torch.cat((x[0],x[1],x[2],x[3],x[4]), dim=1)
            
        
                
        # FOR CNN BASED IMPLEMENTATION
        out = self.pool1(self.relu(self.conv1(x)))
        out = out[:,:int(self.shrink*out.size(1)),:,:] 
        
        out = self.layer1(out)
        out = self.layer2(out)
        #out = self.layer3(out)
        #out = self.layer4(out)
        out = self.pool2(self.relu(self.conv2(out)))
        
        x = out.view(out.size(0), -1)
        x = x[:,:int(self.shrink*10240)] 
        
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.out(x)  # no softmax: CrossEntropyLoss()
        return x
    
class ResLike_(nn.Module):
    def __init__(self, block, num_classes=10):
        super(ResLike, self).__init__()
        self.conv1 = nn.Conv2d(15, 32, kernel_size=(7,7), padding="same")
        self.layer1 = nn.Sequential(block(32, 32, pool_kernel=2))
        self.layer2 = nn.Sequential(block(32, 32, pool_kernel=3, pool_padding=1))
        self.layer3 = nn.Sequential(block(32, 32, pool_kernel=3, pool_padding=1))
        self.layer4 = nn.Sequential(block(32, 32, pool_kernel=3, pool_padding=1))
        
        self.hidden1 = nn.Linear(4160, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, num_classes)  # 128
        #######################
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(32)
    
    def forward(self, *x):
        if isinstance(x, tuple) and len(x)>1:
            x = torch.cat((x[0],x[1],x[2],x[3],x[4]), dim=1)
        # FOR CNN BASED IMPLEMENTATION
        out = self.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        x = out.view(out.size(0), -1)
        
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.out(x)  # no softmax: CrossEntropyLoss()
        return x
        


def EscFusion(conv_layer, bn_layer, **kwargs):
    return ResLike(BasicBlock, num_classes=kwargs['num_classes'], mode=kwargs['mode'])
