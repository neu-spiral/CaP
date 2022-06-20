from .trainer import *
from ..utils.dataset import *
from ..utils.io import *
from ..utils.eval import *
from ..utils.misc import *
from ..utils.masks import *
from ..utils.calculate_flops.calflops import calflops
from .admm import *

import time
from torch.autograd import Variable
from torchsummary import summary

class MoP:
    """
    """

    def __init__(self, configs, print_model=True, print_params=True):
        
        torch.manual_seed(configs['seed'])
        self.configs = configs
        self.model_file = configs['model_file']
        
        # Handle dataset
        self.train_loader, self.test_loader = get_dataset_from_code(configs['data_code'], configs['batch_size'])

        # Initialize evaluation metrics
        self.evalHelper   = EvalHelper(configs['data_code'])
        
        # Load device
        self.device = configs["device"]
        
        # Create model
        self.model = get_model_from_code(configs).to(self.device)
        
        # Load pretrained weights
        if 'load_model' in configs:
            state_dict = torch.load(get_model_path("{}".format(configs["load_model"])), map_location=self.device)
            self.model = load_state_dict(self.model, 
                                         state_dict['model_state_dict'] if 'model_state_dict' in state_dict 
                                         else state_dict['state_dict'] if 'state_dict' in state_dict else state_dict,)
            criterion,_,_ = set_optimizer(self.configs, self.model, self.train_loader, self.configs['optimizer'], 
                                          self.configs['learning_rate'], self.configs['epochs'])
            acc = self.test_model(self.model, criterion)
        else:
            print('standard train')
            #self.model.apply(init_weights)
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
        
        # Get input shape from data_code
        self.input_var = get_input_from_code(configs)
        
        # Config partitions and prune_ratio
        self.configs = partition_generator(configs, self.model)
            
        # Compute output size of each layer
        self.configs['partition'] = featuremap_summary(self.model, self.configs['partition'], self.input_var)
        
        # Setup communication costs
        self.configs['comm_costs'] = set_communication_cost(self.model, self.configs['partition'],)
        
        # Calculate flops
        calflops(self.model, self.input_var)
        
        # Test before prune
        test_partition(self.model, partition=self.configs['partition'])
        
    def prune(self):
        nepoch = self.configs['epochs']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        
        # Initializing ADMM; if not admm, do hard pruning only
        admm = ADMM(self.configs, self.model, rho=self.configs['rho']) if self.configs['admm'] else None

        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, 
                               criterion, optimizer, scheduler, ADMM=admm, comm=True)
            acc = self.test_model(self.model, criterion, cepoch)
            
        # hard prune
        hard_prune(admm, self.model, self.configs['sparsity_type'], option=None)
        
        # test sparsity
        if self.configs['sparsity_type']=='kernel':
            test_kernel_sparsity(self.model, partition=self.configs['partition'])
            test_partition(self.model, partition=self.configs['partition'])
        else:
            test_filter_sparsity(self.model)
        
        # plot first conv layer
        plot_layer(self.model, self.configs['partition'], layer_id=10,
                   savepath=get_fig_path("{}".format('.'.join(self.model_file.split('.')[:-1]))))
        #save_model(self.model, get_model_path("{}.pt".format('.'.join(self.model_file.split('.')[:-1])+'_hardprune')))
                
    def finetune(self):
        # Todo: seperate BN
        #self.parmodel = models.__dict__[self.configs['model']](self.cl, self.bn,
        #                                               num_classes=self.configs['num_classes'],
        #                                               #bn_partition=self.configs['partition']['bn_partition']
        #                                               ).to(self.device)
        
        #self.parmodel = load_state_dict(self.parmodel, 
        #                                self.model.state_dict(),
        #                                #bn_par=True, 
        #                                #partition=self.configs['partition']
        #                                )
        
        print("======== MODEL INFO =========")
        print(self.model)
        print("=" * 40)
        calflops(self.model, self.input_var, self.configs['prune_ratio'])
        
        # get mask
        masks = get_model_mask(model=self.model)
    
        # masked retrain
        nepoch = self.configs['retrain_ep']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, 
                               criterion, optimizer, scheduler, masks=masks)
            acc = self.test_model(self.model, criterion, cepoch)
            if acc > best:
                best = acc
                save_model(self.model, get_model_path("{}".format(self.model_file)))
        
        if self.configs['sparsity_type']=='kernel':
            test_kernel_sparsity(self.model, partition=self.configs['partition'])
            test_partition(self.model, partition=self.configs['partition'])
        else:
            test_filter_sparsity(self.model)
    
    def pruneMask(self):
        nepoch = self.configs['epochs']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        
        # Initializing ADMM; if not admm, do hard pruning only
        admm = ADMM(self.configs, self.model, rho=self.configs['rho'], target='mask') if self.configs['admm'] else None
        
        # fix weights
        set_trainable_mask(self.model, requires_grad=False, target='weight')
        
        # prune
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, criterion, 
                               optimizer, scheduler, ADMM=admm, comm=True)
            acc = self.test_model(self.model, criterion, cepoch)
            
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
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model_r, self.train_loader, \
                                             self.configs['retrain_opt'], self.configs['retrain_lr'], nepoch)
    
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model_r, self.train_loader, 
                               criterion, optimizer, scheduler, masks=masks)
            acc = self.test_model(self.model_r, criterion, cepoch)
            if acc > best:
                best = acc
                save_model(self.model_r, get_model_path("{}".format(self.model_file)))
        
        test_filter_sparsity(self.model_r)
            
    def train(self):
        nepoch = self.configs['epochs']
        criterion, optimizer, scheduler = set_optimizer(self.configs, self.model, self.train_loader, \
                                             self.configs['optimizer'], self.configs['learning_rate'], nepoch)
        best = 0
        for cepoch in range(0, nepoch+1):
            if cepoch>0:
                print('Learning rate: {:.4f}'.format(get_lr(optimizer)))
                standard_train(self.configs, cepoch, self.model, self.train_loader, criterion, optimizer, scheduler)
                
            acc = self.test_model(self.model, criterion, cepoch)
            if acc > best:
                best = acc
                save_model(self.model, get_model_path("{}".format(self.model_file.split('.')[0]+'.pt')))
    
    def test_model(self, model, criterion, cepoch=0):
        acc = self.evalHelper.get_accuracy(model, self.test_loader, criterion, cepoch)
        return acc
    