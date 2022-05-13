import os
import yaml
import torch
import numpy as np

def partition_generator(configs, model):
    partition = {}
    
    # get # partition for each layer
    if configs['num_partition'].isdigit():
        num_partition = {}
        # Todo: automatically set bn_partition
        bn_partition = [int(configs['num_partition'])] * 9
        for name, W in model.named_parameters():
            if (len(W.size()) == 4): num_partition[name] = int(configs['num_partition'])
    elif os.path.exists(configs['num_partition']):
        with open(configs['num_partition'], "r") as stream:
            raw_dict = yaml.safe_load(stream)
            num_partition = raw_dict['partitions']
            bn_partition = raw_dict['bn_partitions']
    else:
        raise Exception("num_partition must be either a filepath or an integer")
    
    # setup bn_partition
    partition['bn_partition'] = bn_partition
    
    # setup prune ratio
    configs['prune_ratio'] = {}
    for name, num in num_partition.items():
        configs['prune_ratio'][name] = 1-1./num
    
    # setup selected kernel ids
    for name, W in model.named_parameters():
        if name in num_partition and num_partition[name] > 1:
            num = num_partition[name]
            k1, m1 = divmod(W.shape[0], num)
            k2, m2 = divmod(W.shape[1], num)
            filter_id, channel_id = [],[]
            for i in range(num):
                filter_id.append(np.array(range(ParCalculator(i,k1,m1),ParCalculator(i+1,k1,m1))))
                channel_id.append(np.array(range(ParCalculator(i,k2,m2),ParCalculator(i+1,k2,m2))))
            partition[name] = {'num': num, 
                               'filter_id': filter_id,
                               'channel_id': channel_id,}
            
            # v0: split by 'jump'
            #own_state[name].copy_(param_s[i::num_partition])
            # v1: split by equal 'cut'
            #len_in, len_out = int(weight.shape[0]/num_partition), int(weight.shape[1]/num_partition)
            #cost_mask[i*len_in:(i+1)*len_in,i*len_out:(i+1)*len_out,:,:] = 0
            # v2: split by app-equal 'cut'
            #k1, m1 = divmod(weight.shape[0], num_partition)
            #k2, m2 = divmod(weight.shape[1], num_partition)
            #for i in range(partition['num']):
            #    cost_mask[i*k1+min(i, m1):(i+1)*k1+min(i+1, m1),i*k2+min(i, m2):(i+1)*k2+min(i+1, m2),:,:] = 0
    configs['partition'] = partition    
    return configs

def ParCalculator(i,k,m):
    return i*k+min(i, m)

def set_communication_cost(model, partition):
    comm_costs = {}
    device = next(model.parameters()).device
    
    for name, W in model.named_parameters():
        if name in partition:
            weight = W.cpu().detach().numpy()
            shape = weight.shape
            cost_mask = np.ones(shape).reshape(shape[0],shape[1], -1)
            
            for i in range(partition[name]['num']):
                cost_mask[partition[name]['filter_id'][i][:,None],partition[name]['channel_id'][i]] = 0
            comm_costs[name] = torch.from_numpy(cost_mask.reshape(shape)).to(device)
            
    return comm_costs

def masknet_to_dense(masknet, model):
    device = next(model.parameters()).device
    own_state = model.state_dict()
    
    # load dense variables
    for (name, W) in masknet.named_parameters():
        if "mask" not in name:
            own_state[name].copy_(W.data)

    # update dense variables
    for (name, W) in masknet.named_parameters():
        if "mask" in name:
            W_d = own_state[name.replace('mask', "weight")]
            
            weight = W.cpu().detach().numpy()
            weight_d = W_d.cpu().detach().numpy()
            
            own_state[name.replace('mask', "weight")].copy_(torch.from_numpy(weight * weight_d))
            
def get_model_mask(model):
    masks = {}
    device = next(model.parameters()).device
    
    for name, W in (model.named_parameters()):
        if 'mask' in name:
            continue
        weight = W.cpu().detach().numpy()
        non_zeros = (weight != 0)
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros)
        W = torch.from_numpy(weight).to(device)
        W.data = W
        masks[name] = zero_mask.to(device)
        #print(name,zero_mask.nonzero().shape)
    return masks

def set_trainable_mask(model, requires_grad=False, target='weight'):
    for name, W in (model.named_parameters()):
        if target in name:
            W.requires_grad = requires_grad