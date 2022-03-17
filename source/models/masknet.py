from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__(in_features, out_features, True)
        self.w_mask = Parameter(torch.ones(out_features, in_features))
        #self.b_mask = Parameter(torch.ones(out_features))

    def forward(self, input):
        return F.linear(input, self.weight*self.w_mask, self.bias)


class MaskConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
    
    def _conv_forward(self, input, weight):
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
        #         padding, dilation, groups, bias, padding_mode)
        self.w_mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
        
        #self.w_mask[1::2,1::2,:,:] = 1
        
    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.w_mask)
