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

        # config partition number
        self.num_partition = configs['num_partition']
        if self.num_partition > 1:
            self.configs['prune_ratio'] = 1-1./self.num_partition
            #print(self.configs['prune_ratio'],self.num_partition,1./self.num_partition)
        
        # Create model
        cl = get_layers(configs['layer_type'])
        bn = get_bn_layers(configs['bn_type'])
        
        self.model = models.__dict__[configs['model']](cl, bn,
                                                       num_classes=configs['num_classes'],
                                                       num_partition=self.num_partition if configs['bn_type']=='masked' else 1
                                                       ).to(self.device)
        print(self.model)
        # Load pretrained weights
        if 'load_model' in configs:
            self.model = load_state_dict(self.model, 
                                         get_model_path("{}".format(configs["load_model"])),
                                         num_partition=self.num_partition,
                                         bn_type=configs['bn_type']
                                         )
        else:
            print('standard train')
            print(self.model)
            self.train()
        
        # set communication costs
        self.configs['comm_costs'] = set_communication_cost(self.model,
                                                            num_partition=self.num_partition,
                                                            target=configs['layer_type'])
        
        
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
        
        # fix weights
        #set_trainable_mask(self.model, requires_grad=False, target='weight')
        
        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler, ADMM=admm, comm=True)
            acc = self.test_model(self.model, cepoch)
            
        # hard prune
        hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
        
        if self.configs['sparsity_type']=='kernel':
            test_kernel_sparsity(self.model)
            test_partition(self.model, num_partition=self.num_partition)
        else:
            test_filter_sparsity(self.model)
        save_model(self.model, get_model_path("{}.pt".format('.'.join(self.model_file.split('.')[:-1])+'_hardprune')))
        
    def finetune(self):
        # get mask
        masks = get_model_mask(model=self.model)
    
        # masked retrain
        nepoch = self.configs['retrain_ep']
        optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler, masks=masks)
            acc = self.test_model(self.model, cepoch)
            if acc > best:
                best = acc
                save_model(self.model, get_model_path("{}".format(self.model_file)))
        
        if self.configs['sparsity_type']=='kernel':
            test_kernel_sparsity(self.model)
            test_partition(self.model, num_partition=self.num_partition)
        else:
            test_filter_sparsity(self.model)
    
    def pruneMask(self):
        nepoch = self.configs['epochs']
        optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        
        # Initializing ADMM; if not admm, do hard pruning only
        admm = ADMM(self.configs, self.model, rho=self.configs['rho'], target='mask') if self.configs['admm'] else None
        
        # fix weights
        set_trainable_mask(self.model, requires_grad=False, target='weight')
        
        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, optimizer, scheduler, ADMM=admm, comm=True)
            acc = self.test_model(self.model, cepoch)
            
        # hard prune
        hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
        test_filter_sparsity(self.model)
            
    def finetuneWeight(self):
        self.model_r = models.__dict__[self.configs['model']](get_layers('regular'), get_bn_layers('regular'),
                                                       num_classes=self.configs['num_classes'],
                                                       ).to(self.device)
        # transfer weight
        print('Finetune starts')
        masknet_to_dense(self.model, self.model_r)
        test_filter_sparsity(self.model_r)
        test_partition(self.model_r, num_partition=self.num_partition)
        
        # get mask
        masks = get_model_mask(model=self.model_r)
    
        # masked retrain
        nepoch = self.configs['retrain_ep']
        optimizer, scheduler = set_optimizer(self.configs, self.model_r, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model_r, self.train_loader, optimizer, scheduler, masks=masks)
            acc = self.test_model(self.model_r, cepoch)
            if acc > best:
                best = acc
                save_model(self.model_r, get_model_path("{}".format(self.model_file)))
        
        test_filter_sparsity(self.model_r)
            
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
                save_model(self.model, get_model_path("{}".format(self.model_file.split('.')[0]+'_teacher'+'.pt')))
    
    def test_model(self, model, cepoch=0):
        criterion = torch.nn.CrossEntropyLoss()
        acc = get_accuracy(model, self.test_loader, criterion, cepoch)
        return acc
    