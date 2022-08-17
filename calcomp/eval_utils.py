import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from wrn import wrn16_8, wrn28_10, wrn28_4
from resnet import resnet18, resnet34, resnet50
from escnet import EscFusion
from flashnet import InfoFusionThree, CoordNet, CameraNet, LidarNet

# Nasim
import time
import statistics

def ModelLoader(dataset, mode=''):
    if dataset=='cifar10':
        model = resnet18(nn.Conv2d, nn.BatchNorm2d, num_classes=10).to(device)
        data = np.random.uniform(0, 1, (1, 3, 32, 32))
    elif dataset=='cifar100':
        model = wrn16_8(nn.Conv2d, nn.BatchNorm2d, num_classes=100).to(device)
        data = np.random.uniform(0, 1, (1, 3, 32, 32))
    elif dataset=='flash':
        num_classes = 64
        model = InfoFusionThree(nb_classes=num_classes, mode=mode).to(device)
        input_shape = [(1, 64),
                       (1, 256),
                       (1, 512),]
        data = ()
        for x in input_shape:
            data += (Variable(torch.FloatTensor(np.random.uniform(0, 1, x))).to(device),)

    elif dataset=='esc':
        model = EscFusion(nn.Conv2d, nn.BatchNorm2d, num_classes=2, mode=mode).to(device)
        input_shape = [(3, 266, 320) for _ in range(5)]
        data = ()
        for x in input_shape:
            data += (Variable(torch.FloatTensor(np.random.uniform(0, 1, (1,)+x))).to(device),)
            
    return model, data


'''
Use cuda
'''
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
writer = None

'''
set up model&inputs
'''
#datasets = ['cifar10', 'cifar100', 'flash', 'esc']
datasets = ['flash','esc',]
models = {'cifar10': 'cifar10-resnet18-kernel-npv2-pr0.75-lcm0.000001.pt', 
          'cifar100': 'cifar100-wrn28_10-kernel-npv3-lcm0.00001-lcp0.pt',
          'flash': 'flash-InfoFusionThree-kernel-np3-pr0.75-lcm0.001.pt', 
          'esc': 'esc-EscFusion-kernel-np5-pr0.85-lcm1.pt'}

for dataset in datasets:
    '''
    estimate computational cost for pre-trained models
    '''
    
    print('this dataset is: ' + str(dataset))
    model, input_var = ModelLoader(dataset=dataset)
    model.eval()
    output =  model(*input_var)
    time_list = []
    for _ in range(1000):   # run the inference 1000 times
        t1 = time.time()
        output = model(*input_var)
        duration = time.time()-t1
        time_list.append(duration)
    mean = sum(time_list)/len(time_list)
    std = statistics.pstdev(time_list)
    print('Mean and standard deviation for the pre-trained model are :')
    print(mean,std,'seconds')
                                               
    print('for prune')
    model, input_var = ModelLoader(dataset=dataset, mode='calcomp')
    model.eval()
    time_list = []
    for _ in range(1000):   # run the inference 1000 times
        t1 = time.time()
        output = model(*input_var)
        duration = time.time()-t1
        time_list.append(duration)
    mean = sum(time_list)/len(time_list)
    std = statistics.pstdev(time_list)
    print('Mean and standard deviation for the pre-trained model are :')
    print(mean,std,'seconds')

    '''
    estimate computational cost for pruned models
    '''
    '''                                         
    model.load_state_dict(torch.load(models[dataset],map_location='cuda:0'))#+'.pt'))
    model.eval()
    
    time_list = []
    for _ in range(1000):   # run the inference 1000 times
        t1 = time.time()
        output = model(input_var)
        duration = time.time()-t1
        time_list.append(duration)
    mean = sum(time_list)/len(time_list)
    std = statistics.pstdev(time_list)
    print('Mean and standard deviation for the pruned model are :')
    print(mean,std,'seconds')
    '''

