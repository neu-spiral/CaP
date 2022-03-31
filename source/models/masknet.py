from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class BatchNorm2dPartition(nn.Module):
    def __init__(self, planes, num_partition=1):
        super(BatchNorm2dPartition, self).__init__()
        self.num_partition = num_partition
        # To do: handle non-divisable
        #self.len_partition = int(planes / num_partition)
        self.k, self.m = divmod(planes, num_partition)
        
        self.bn_list = nn.ModuleList(nn.BatchNorm2d(self.k+int(i<self.m)) for i in range(num_partition))
        
    def forward(self, x):
        #out_list = [bn(x[:,i::self.num_partition,:,:]) for i, bn in enumerate(self.bn_list)]
        #for i in range(self.num_partition):
        #    x[:,i::self.num_partition,:,:] = out_list[i]
        #return x
    
        #[i*k+min(i, m):(i+1)*k+min(i+1, m)]
        out_list = [bn(x[:,i*self.k+min(i,self.m):(i+1)*self.k+min(i+1, self.m),:,:]) for i, bn in enumerate(self.bn_list)]
        out = torch.cat(out_list, axis=1)
        return out
        
    
class MaskLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super(MaskLinear, self).__init__(in_features, out_features, True)
        self.mask = Parameter(torch.ones(out_features, in_features))
        #self.b_mask = Parameter(torch.ones(out_features))

    def forward(self, input):
        return F.linear(input, self.weight*self.mask, self.bias)


class MaskConv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.mask = Parameter(torch.ones(self.weight.shape))
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
    
    def _conv_forward(self, input, weight):
        return F.conv1d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * self.mask)

class MaskConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        #super(MaskConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
        #         padding, dilation, groups, bias, padding_mode)
        self.mask = Parameter(torch.ones(self.weight.shape)*3)
        #nn.init.kaiming_uniform_(self.mask)
        #self.b_mask = Parameter(torch.ones(self.bias.shape))
        
        #self.w_mask[1::2,1::2,:,:] = 1
        
    def _conv_forward(self, input, weight):
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight * F.hardsigmoid(self.mask))
