import time
from tqdm import tqdm
from ..utils.misc import *
from ..utils.eval import *
from .admm import *

def standard_train(configs, cepoch, model, data_loader, optimizer, scheduler, ADMM=None, masks=None, comm=False):

    batch_acc    = AverageMeter()
    batch_loss   = AverageMeter()
    batch_comm   = AverageMeter()
    
    n_data = configs['batch_size'] * len(data_loader)
    
    if ADMM is not None: 
        admm_initialization(configs, ADMM=ADMM, model=model)
        
    start_time = time.time()
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    for batch_idx, (data, target) in pbar:
           
        data   = data.to(configs['device'])
        target = target.to(configs['device'])
        total_loss = 0
        comm_loss = 0
            
        optimizer.zero_grad()
        
        if configs['mixup']:
            data, target_a, target_b, lam = mixup_data(data, target, configs['alpha'])
        
        output = model(data)
        criterion = CrossEntropyLossMaybeSmooth(smooth_eps=configs['smooth_eps']).to(configs['device'])
        if configs['mixup']:
            loss = mixup_criterion(criterion, output, target_a, target_b, lam, configs['smooth'])
        else:
            loss = criterion(output, target, smooth=configs['smooth'])
        total_loss += (loss * configs['xentropy_weight'])
        
        if ADMM is not None:
            z_u_update(configs, ADMM, model, cepoch, batch_idx)  # update Z and U variables
            prev_loss, admm_loss, total_loss = append_admm_loss(ADMM, model, total_loss)  # append admm losses
            
        if comm:
            for (name, W) in model.named_parameters():
                if name in ADMM.prune_ratios:
                    comm_cost = torch.abs(W) * configs['comm_costs'][name]
                    comm_loss += comm_cost.view(comm_cost.size(0), -1).sum()
            total_loss += configs['lambda_comm'] * comm_loss
        
        total_loss.backward() # Back Propagation
        
        # For masked training
        if masks is not None:
            with torch.no_grad():
                for name, W in (model.named_parameters()):
                    if name in masks and W.grad is not None:
                        W.grad *= masks[name]
                        
        optimizer.step()
        
        # adjust learning rate
        if ADMM is not None:
            admm_adjust_learning_rate(optimizer, cepoch, configs)
        else:
            scheduler.step()

        acc1 = accuracy(output, target, topk=(1,))
        batch_loss.update(loss.item(), data.size(0))
        batch_comm.update(comm_loss.item() if comm_loss else comm_loss, data.size(0))
        batch_acc.update(acc1[0].item(), data.size(0))

        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] Loss:{loss:.4f} CommLoss:{commloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data), 
                        loss = batch_loss.avg,
                        commloss = batch_comm.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))

    
def distill_train(configs, cepoch, teacher, student, data_loader, optimizer, scheduler):

    batch_acc    = AverageMeter()
    distillloss   = AverageMeter()
    filtloss   = AverageMeter()
    
    n_data = configs['batch_size'] * len(data_loader)
    
    start_time = time.time()
    pbar = tqdm(enumerate(data_loader), total=n_data/configs['batch_size'], ncols=150)
    for batch_idx, (data, target) in pbar:
           
        data   = data.to(configs['device'])
        target = target.to(configs['device'])
        
        optimizer.zero_grad()
        
        t_output, t_filt = teacher(data)
        s_output, s_filt = student(data)
        distill_loss = distillation(s_output, t_output, target, configs['distill_temp'], configs['distill_alpha'],)
        filt_loss = sum([actTransfer_loss(x, y) for x, y in zip([s_filt], [t_filt])])
        loss = distill_loss + configs['lambda_f']*filt_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1 = accuracy(s_output, target, topk=(1,))
        distillloss.update(distill_loss.item(), data.size(0))
        filtloss.update(filt_loss.item(), data.size(0))
        batch_acc.update(acc1[0].item(), data.size(0))

        # # # preparation log information and print progress # # #
        msg = 'Train Epoch: {cepoch} [ {cidx:5d}/{tolidx:5d} ({perc:2d}%)] DistillLoss:{distillloss:.4f} FiltLoss:{filtloss:.4f} Acc:{acc:.4f}'.format(
                        cepoch = cepoch,  
                        cidx = (batch_idx+1)*configs['batch_size'], 
                        tolidx = n_data,
                        perc = int(100. * (batch_idx+1)*configs['batch_size']/n_data),
                        distillloss = distillloss.avg,
                        filtloss = filtloss.avg,
                        acc  = batch_acc.avg,
                    )

        pbar.set_description(msg)
    #print('Training time per epoch is {:.2f}s.'.format(time.time()-start_time))
    
