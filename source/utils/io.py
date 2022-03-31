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

def load_state_dict(model, filepath, num_partition=1, bn_type='regular'):
    device = next(model.parameters()).device
    own_state = model.state_dict()
    
    state_dict = torch.load(filepath, map_location=device)
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    
    
    # handle batch norm
    if bn_type=='masked':
        for name, param in own_state.items():
            if 'bn' in name:
                if 'tracked' in name: continue
                param = param.data
                namepiece = name.split('.')
                try: i = int(namepiece[-2]) 
                except: continue

                name_s = '.'.join(namepiece[:-3])+'.'+namepiece[-1]
                if name_s not in state_dict: 
                    continue
                
                param_s = state_dict[name_s].data
                #own_state[name].copy_(param_s[i::num_partition])
                k, m = divmod(param_s.shape[0], num_partition)
                own_state[name].copy_(param_s[i*k+min(i,m) : (i+1)*k+min(i+1, m)])
    
    # for others
    for name, param in state_dict.items():
        if name not in own_state:
            #print('not found: ',name)
            continue
        param = param.data
        own_state[name].copy_(param)
    #print(state_dict['layer1.0.bn1.weight'])
    return model

#def load_bn(param, num_partition=1):
    
def get_model_path(filename, idx=None):
    filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath