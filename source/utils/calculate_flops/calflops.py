from __future__ import print_function
import torchvision.models as models
from collections import OrderedDict
import torch
import argparse
import os
import sys
import yaml

sys.path.append('/home/tong/Model_Partition/MoP/source/utils/calculate_flops/')
from ptflops.flops_counter import get_model_complexity_info
from thop import profile


def calflops(model, inputs, prune_ratios=[]): 
    
        
    if not prune_ratios:
        prune_ratios = OrderedDict()
        with torch.no_grad():
            for name, W in (model.named_parameters()):
                prune_ratios[name] = 0
    
    model.train(False)
    model.eval()
    macs, params = profile(model, inputs=(inputs, ), rate = prune_ratios)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs * 2/1000000000)) # GMACs
    print('{:<30}  {:<8}'.format('Number of parameters: ', params/1000000)) # M
    #flops, params = get_model_complexity_info(model, (3, 32, 32), as_strings=True, print_per_layer_stat=False)
    #print(flops, params)