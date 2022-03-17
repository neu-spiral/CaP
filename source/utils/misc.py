import yaml
import math
import operator
import numpy as np
from numpy import linalg as LA
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

from ..models.masknet import *

def get_layers(layer_type):
    """
    Returns: (conv_layer)
    """
    if layer_type == "regular":
        return nn.Conv2d
    elif layer_type == "masked":
        return MaskConv2d
    else:
        raise ValueError("Incorrect layer type")

        
def set_optimizer(configs, model, train_loader, opt, lr, epochs):
    """ bag of tricks set-ups"""
    configs['smooth'] = configs['smooth_eps'] > 0.0
    configs['mixup'] = configs['alpha'] > 0.0

    optimizer_init_lr = configs['warmup_lr'] if configs['warmup'] else lr
    if opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), optimizer_init_lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), optimizer_init_lr)
    
    scheduler = None
    if configs['lr_scheduler'] == 'cosine':
        print('using cosine')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(train_loader), eta_min=4e-08)
    else:
        """Set the learning rate of each parameter group to the initial lr decayed
                by gamma once the number of epoch reaches one of the milestones
        """
        gamma=0.1
        if configs['data_code'] == 'mnist':
            if epochs <= 50:
                epoch_milestones = [20, 40]
            else:
                epoch_milestones = [75, 90]
        elif configs['data_code'] == 'cifar10':
            if epochs > 150:
                epoch_milestones = [80, 150]
            else:
                epoch_milestones = [65, 90]
        elif configs['data_code'] == 'cifar100':
            # Adversarial Concurrent Training: Optimizing Robustness and Accuracy Trade-off of Deep Neural Networks
            if epochs > 150:
                epoch_milestones = [80, 150]
            else:
                epoch_milestones = [65, 90]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * len(train_loader) for i in epoch_milestones], gamma=gamma)
    #print(epochs, [i * len(train_loader) for i in epoch_milestones])
    if configs['warmup']:
        scheduler = GradualWarmupScheduler(optimizer,multiplier=configs['learning_rate']/configs['warmup_lr'],
                                           total_iter=configs['warmup_epochs'] * len(train_loader),
                                           after_scheduler=scheduler)
    return optimizer, scheduler

def distillation(student_scores, teacher_scores, labels, temperature, alpha):
    criterion = torch.nn.KLDivLoss(log_target=True)
    p_t = F.log_softmax(teacher_scores / temperature, dim=1)
    p_s = F.log_softmax(student_scores / temperature, dim=1)
    h = F.cross_entropy(student_scores, labels)
    return criterion(p_s, p_t) * (temperature * temperature * 2. * alpha) + h * (1. - alpha)

def actTransfer_loss(x, y, normalize_acts=True):
    if normalize_acts:
        return (F.normalize(x.view(x.size(0), -1)) - F.normalize(y.view(y.size(0), -1))).pow(2).mean()
    else:
        return (x.view(x.size(0), -1) - y.view(y.size(0), -1)).pow(2).mean()  
    
class CrossEntropyLossMaybeSmooth(nn.CrossEntropyLoss):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    def __init__(self, smooth_eps=0.0):
        super(CrossEntropyLossMaybeSmooth, self).__init__()
        self.smooth_eps = smooth_eps

    def forward(self, output, target, smooth=False):
        if not smooth:
            return F.cross_entropy(output, target)

        target = target.contiguous().view(-1)
        n_class = output.size(1)
        one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
        smooth_one_hot = one_hot * (1 - self.smooth_eps) + (
            1 - one_hot) * self.smooth_eps / (n_class - 1)
        log_prb = F.log_softmax(output, dim=1)
        loss = -(smooth_one_hot * log_prb).sum(dim=1).mean()
        return loss


def mixup_data(x, y, alpha=1.0):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam, smooth):
    return lam * criterion(pred, y_a, smooth=smooth) + \
           (1 - lam) * criterion(pred, y_b, smooth=smooth)


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_iter: target learning rate is reached at total_iter, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self,
                 optimizer,
                 multiplier,
                 total_iter,
                 after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.total_iter = total_iter
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_iter:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_iter) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_iter + 1.) for base_lr in self.base_lrs]
        

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']