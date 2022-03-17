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

def load_state_dict(model, filepath):
    device = next(model.parameters()).device
    own_state = model.state_dict()
    
    state_dict = torch.load(filepath, map_location=device)
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    
    for name, param in state_dict.items():
        if name not in own_state:
            print('not found: ',name)
            continue
        param = param.data
        own_state[name].copy_(param)
    return model

def get_model_path(filename, idx=None):
    filepath = "{}/assets/models/{}".format(os.getcwd(), filename)
    return filepath