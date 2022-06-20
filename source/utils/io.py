import os
import yaml
import torch

def load_yaml(filepath):

    with open(filepath, 'r') as stream:
        try:
            data = yaml.load(stream, yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return data
    
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    
def load_model(filepath):
    model = torch.load(filepath)
    return model

def load_state_dict(model, state_dict, bn_par=False, partition={}):
    own_state = model.state_dict()
    # handle batch norm
    if bn_par:
        for name, param in own_state.items():
            if 'bn' in name:
                if 'tracked' in name: continue
                
                param = param.data
                namepiece = name.split('.')
                try: i = int(namepiece[-2]) 
                except: continue

                # get layer name of the source model
                name_s = '.'.join(namepiece[:-3])+'.'+namepiece[-1]
                if name_s not in state_dict: 
                    continue
                param_s = state_dict[name_s].data
                
                # get layer name of the corresponding conv layer
                name_par = name_s.replace('bn','conv')
                name_par = name_par.replace('bias','weight')
                name_par = name_par.replace('running_mean','weight')
                name_par = name_par.replace('running_var','weight')
                name_par = name_par.replace('shortcut.1','shortcut.0')
                
                #print(name, param.shape, partition)
                #k, m = divmod(param_s.shape[0], partition)
                #own_state[name].copy_(param_s[i*k+min(i,m) : (i+1)*k+min(i+1, m)])
                own_state[name].copy_(param_s[partition[name_par]['filter_id'][i]])
    
    # for others
    for name, param in state_dict.items():
        name = name.replace('module.','')
        if name not in own_state:
            print('not found: ',name)
            continue
        param = param.data
        own_state[name].copy_(param)
    #print(state_dict['layer1.0.bn1.weight'])
    return model

#def load_bn(param, num_partition=1):
    
def get_model_path(filename, idx=None):
    filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath

def get_fig_path(filename, idx=None):
    filepath = "{}/assets/figs/{}".format(os.getcwd(), filename)
    return filepath