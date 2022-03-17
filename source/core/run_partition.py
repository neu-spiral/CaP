from .. import *
from . import *
from .engine import *

def get_args():
    """ args from input
    """
    parser = argparse.ArgumentParser(description='Model Partition')
    
    parser.add_argument('-cfg', '--config', required=True,type=str, help='config input path')
    parser.add_argument('-tt', '--training-type', default='',type=str, help='training types [hsicprune|backprop]')
    parser.add_argument('-bs', '--batch-size', default=0,type=int, help='minibatch size')
    parser.add_argument('-op', '--optimizer', default='', type=str, help='optimizer')
    parser.add_argument('-lr', '--learning-rate', default=0,type=float, help='learning rate')
    parser.add_argument('-ep', '--epochs', default=-1,type=int, help='number of training epochs')
    parser.add_argument('-sd', '--seed', default=0,type=int, help='random seed for the trial')
    parser.add_argument('-dc', '--data-code', default='',type=str, help='name of the working dataset [mnist|cifar10|cifar100]')
    parser.add_argument('-m', '--model', default='',type=str, help='model architecture')
    parser.add_argument('-mf', '--model-file', default='',type=str, help='filename for saved model file')
    parser.add_argument('-nc', '--num_classes', default=0, type=int, help='number of classes')
    parser.add_argument('--device', type=str, default='',help='CUDA training')

    ### arguments for pruning
    parser.add_argument('-st', '--sparsity-type', default='', type=str, help='for pruning')
    parser.add_argument('-pr', '--prune-ratio', default=1, type=float, help='pruning ratio')
    parser.add_argument('--admm', action='store_true', help='prune by admm')
    parser.add_argument('--admm-epochs', default=0, type=int, help='number of interval epochs to update admm (default: 1)')
    parser.add_argument('--rho', default=0, type=float, help='admm learning rate (default: 1)')
    parser.add_argument('--multi_rho', action='store_true', help='It works better to make rho monotonically increasing')
    parser.add_argument('-ree', '--retrain-ep', default=-1, type=int, help='training epoch of the masked retrain (default: -lr)')
    parser.add_argument('-relr', '--retrain-lr', default=0, type=float, help='learning rate of the masked retrain')
    parser.add_argument('-rebs', '--retrain-bs', default=0, type=int, help='batch size of the masked retrain (default: -bs)')
    parser.add_argument('-relx', '--retrain-lx', default=0, type=float, help='lx of the masked retrain (default: -bs)')
    parser.add_argument('-rely', '--retrain-ly', default=0, type=float, help='ly of the masked retrain (default: -bs)')
    parser.add_argument('-reopt', '--retrain-opt', default='',type=str, help='retraining optimizer')
    parser.add_argument('-rett', '--retraining-type', default='',type=str, help='retraining types [hsictprune|backprop]')
    parser.add_argument('-slmo', '--save_last_model_only', action='store_true', help='save last model only')
    parser.add_argument('-lm', '--load-model', default='', type=str, help='filename of the pre-trained model file')
    
    # control the weight of xentropy and hsic, if not specified in yaml file
    parser.add_argument('-xw', '--xentropy-weight', default=0,type=float, help='how much weight to put on xentropy wrt hsic')
    
    
    ### Tricks for cifar10 but not used:
    parser.add_argument('--lr-scheduler', type=str, default='', help='define lr scheduler')
    parser.add_argument('--warmup', action='store_true', default=False, help='warm-up scheduler')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='M', help='warmup-lr, smaller than original lr')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='M', help='number of epochs for lr warmup')
    parser.add_argument('--mixup', action='store_true', default=False, help='ce mixup')
    parser.add_argument('--alpha', type=float, default=0.0, metavar='M', help='for mixup training, lambda = Beta(alpha, alpha) distribution. Set to 0.0 to disable')
    parser.add_argument('--smooth', action='store_true', default=False, help='lable smooth')
    parser.add_argument('--smooth-eps', type=float, default=0.0, metavar='M', help='smoothing rate [0.0, 1.0], set to 0.0 to disable')
    
    # arguments for distillation
    parser.add_argument('--distill_model', type=str, default='', help='where to load pretrained distillation model')
    parser.add_argument('--distill-loss', type=str, default='kl', help='type of distillation loss')
    parser.add_argument('--distill-temp', default=30, type=float, help='temperature for kl')
    parser.add_argument('--distill-alpha', default=1, type=float, help='weight for distillation loss')
   
    # partition
    parser.add_argument('-lt', '--layer-type', default='', type=str, help='regular/masked')
    parser.add_argument('--num-students', type=int, default=0, help='number of students')
    parser.add_argument('--filter-sizes', metavar='N', default='', help ="Ex, for 2 students --filter_sizes 64,64")
    parser.add_argument('-lf', '--lambda_f', default=0, type=float, help='the coefficient of the HSIC objective')  
    args = parser.parse_args()

    return args
    
def main():

    args = get_args()
    config_dict = load_yaml(args.config)
    
    # gpu device
    config_dict['device'] = args.device
    
    # partition
    if args.layer_type:
        config_dict['layer_type'] = args.layer_type
    
    # distillation
    config_dict['distill_model'] = args.distill_model
    config_dict['distill_loss'] = args.distill_loss
    config_dict['distill_temp'] = args.distill_temp
    config_dict['distill_alpha'] = args.distill_alpha

    if args.optimizer:
        config_dict['optimizer'] = args.optimizer
    if args.seed:
        config_dict['seed'] = args.seed
    if args.learning_rate:
        config_dict['learning_rate'] = args.learning_rate
    if args.model_file:
        config_dict['model_file'] = args.model_file
    if args.load_model:
        config_dict['load_model'] = args.load_model
    if args.batch_size:
        config_dict['batch_size'] = args.batch_size
    if args.epochs >= 0:
        config_dict['epochs'] = args.epochs
    if args.data_code:
        config_dict['data_code'] = args.data_code
    if args.model:
        config_dict['model'] = args.model
    if 'num_classes' not in config_dict:
        config_dict['num_classes'] = args.num_classes
   
    # admm prune
    if args.admm or 'admm' not in config_dict:
        config_dict['admm'] = args.admm
    if args.admm_epochs:
        config_dict['admm_epochs'] = args.admm_epochs
    if args.sparsity_type:
        config_dict['sparsity_type'] = args.sparsity_type
    if args.prune_ratio:
        config_dict['prune_ratio'] = args.prune_ratio
    if args.rho:
        config_dict['rho'] = args.rho
    if args.multi_rho:
        config_dict['multi_rho'] = args.multi_rho

    # finetune
    if args.retrain_lr:
        config_dict['retrain_lr'] = args.retrain_lr
    if args.retrain_bs:
        config_dict['retrain_bs'] = args.retrain_bs
    if args.retrain_ep >= 0:
        config_dict['retrain_ep'] = args.retrain_ep
    if args.retrain_lx:
        config_dict['retrain_lx'] = args.retrain_lx
    if args.retrain_ly:
        config_dict['retrain_ly'] = args.retrain_ly
    if args.retrain_opt:
        config_dict['retrain_opt'] = args.retrain_opt
    if args.retraining_type:
        config_dict['retraining_type'] = args.retraining_type

    ## comparison with light model train from scratch
    config_dict['save_last_model_only'] = args.save_last_model_only
    
    if args.xentropy_weight:
        config_dict['xentropy_weight'] = args.xentropy_weight
        
    # tricks:
    if 'lr_scheduler' not in config_dict:
        config_dict['lr_scheduler'] = 'cosine'
    if args.lr_scheduler:
        config_dict['lr_scheduler'] = args.lr_scheduler
    if args.warmup or 'warmup' not in config_dict:
        config_dict['warmup'] = args.warmup
    if 'warmup_lr' not in config_dict:
         config_dict['warmup_lr'] = args.warmup_lr 
    if 'warmup_epochs' not in config_dict:
        config_dict['warmup_epochs'] = args.warmup_epochs 
    if 'mix_up' not in config_dict:
        config_dict['mix_up'] = False 
    if 'alpha' not in config_dict:
        config_dict['alpha'] = 0 
    if 'smooth' not in config_dict:
        config_dict['smooth'] = False 
    if 'smooth_eps' not in config_dict:
        config_dict['smooth_eps'] = 0 
    
    for key, val in config_dict.items():
        print(key, ': ', val)
    return config_dict
   
    
if __name__ == "__main__":
    configs = main()
    print(configs)
    mop = MoP(configs)
    mop.prune()
    mop.finetune()
   