import torch
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

class AverageMeter(object):
    """Basic meter"""
    def __init__(self):
        self.reset()

    def reset(self):
        """ reset meter
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ incremental meter
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EvalHelper():
    def __init__(self, data_code):
        self.data_code = data_code
    
    def call(self, output, target):
        if self.data_code == 'flash':
            acc = [(torch.argmax(target, axis=1) == torch.argmax(output, axis=1)).sum()*100/target.size(0)]
        else:
            acc = self.accuracy(output, target, topk=(1,))
        return acc
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def get_accuracy(self, model, dataloader, criterion, cepoch):
        """ Computes the precision@k for the specified values of k
            https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        losses = AverageMeter()
        top1 = AverageMeter()
        device = next(model.parameters()).device

        # for binary case
        output_all, target_all = [], []
        softmax = torch.nn.Softmax(dim=1)
        
        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                data   = ()
                for piece in batch[:-1]:
                    data += (piece.float().to(device),)
                target = batch[-1].to(device)

                # compute output
                output = model(*data)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1 = self.call(output, target)
                losses.update(loss.item(), target.size(0))
                top1.update(acc1[0].item(), target.size(0))
                
                # only measured for binary classification
                if self.data_code == 'esc':
                    output = softmax(output).cpu().detach().numpy()
                    target = target.cpu().detach().numpy()
                    output_all = np.vstack((output_all,output)) if len(output_all) else output
                    target_all = np.vstack((target_all,target[:,None])) if len(target_all) else target[:,None]
                    
        print("Epoch-[{:03d}]: Test loss: {:.2f}, acc: {:.2f}.".format(cepoch, losses.avg, top1.avg,))
        
        if self.data_code == 'esc':
            auc = roc_auc_score(target_all, output_all[:, 1])
            prec, recall, fscore, _ = precision_recall_fscore_support(target_all, np.argmax(output_all, axis=1), average='macro')
            print("prec: {:.4f}, recall: {:.4f}, auc: {:.4f}".format(prec, recall, auc))
        

        return top1.avg