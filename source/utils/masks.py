import torch
import numpy as np

def set_communication_cost(model,num_partition,target='weight'):
    comm_costs = {}
    device = next(model.parameters()).device
    
    for name, W in model.named_parameters():
        if (len(W.size()) == 4) and (target in name):
            if name == 'conv1.'+ target: continue
            weight = W.cpu().detach().numpy()
            cost_mask = np.ones(weight.shape)
            #len_in, len_out = int(weight.shape[0]/num_partition), int(weight.shape[1]/num_partition)
            #cost_mask[i*len_in:(i+1)*len_in,i*len_out:(i+1)*len_out,:,:] = 0
            k1, m1 = divmod(weight.shape[0], num_partition)
            k2, m2 = divmod(weight.shape[1], num_partition)
            for i in range(num_partition):
                cost_mask[i*k1+min(i, m1):(i+1)*k1+min(i+1, m1),i*k2+min(i, m2):(i+1)*k2+min(i+1, m2),:,:] = 0
            comm_costs[name] = torch.from_numpy(cost_mask).to(device)
            
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