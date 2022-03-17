import models
from .trainer import *
from ..utils.dataset import *
from ..utils.io import *
from ..utils.eval import *
from ..utils.misc import *
from ..utils.masks import *
from .admm import *

class MoP:
    """
    """

    def __init__(self, configs, print_model=True, print_params=True):
        
        torch.manual_seed(configs['seed'])
        self.configs = configs
        self.model_file = configs['model_file']
        # Handle dataset
        self.train_loader, self.test_loader = get_dataset_from_code(configs['data_code'], configs['batch_size'])

        # Load device
        self.device = configs["device"]

        # Create model
        cl = get_layers(configs['layer_type'])
        
        self.model = models.__dict__[configs['model']](cl, num_classes=configs['num_classes']).to(self.device)
        
        # Load pretrained weights
        if 'load_model' in configs:
            self.model = load_state_dict(self.model, get_model_path("{}".format(configs["load_model"])))
        else:
            print('standard train')
            print(self.model)
            self.train()
         
        # Print the model
        if print_model:
            print("======== MODEL INFO =========")
            print(self.model)
            print("=" * 40)

        # Print the number of parameters
        if print_params:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"TOTAL NUMBER OF PARAMETERS = {n_parameters}")
            print("-" * 40)
    
    def prune(self):
        nepoch = self.configs['epochs']
        optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        
        # Initializing ADMM; if not admm, do hard pruning only
        admm = ADMM(self.configs, self.model, rho=self.configs['rho']) if self.configs['admm'] else None
        
        # fix mask
        #set_trainable_mask(self.model, requires_grad=True)
        
        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler, ADMM=admm)
                acc = self.test_model(cepoch)
        
        # hard prune
        hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
        
        if self.configs['sparsity_type']=='filter' or self.configs['sparsity_type']=='partition':
            test_filter_sparsity(self.model)
        else:
            test_irregular_sparsity(self.model)
         
    def finetune(self):
        # get mask
        masks = get_model_mask(model=self.model)
    
        # masked retrain
        nepoch = self.configs['retrain_ep']
        optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
            standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler, masks=masks)
            acc = self.test_model(cepoch)
            if acc > best:
                best = acc
                save_model(self.model, get_model_path("{}".format(self.model_file)))
        
        if self.configs['sparsity_type']=='filter':
            test_filter_sparsity(self.model)
        else:
            test_irregular_sparsity(self.model)
            
    def train(self):
        nepoch = self.configs['epochs']
        optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        best = 0
        for cepoch in range(0, nepoch+1):
            print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
            standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler)
            acc = self.test_model(cepoch)
            if acc > best:
                best = acc
                save_model(self.model, get_model_path("{}".format(self.model_file+'_teacher')))
    
    def test_model(self, cepoch=0):
        criterion = torch.nn.CrossEntropyLoss()
        acc = get_accuracy(self.model, self.test_loader, criterion, cepoch)
        return acc
    