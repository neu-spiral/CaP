import torch

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

    
def accuracy(output, target, topk=(1,)):
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
    
def get_accuracy(model, dataloader, criterion, cepoch):
    """ Computes the precision@k for the specified values of k
        https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    losses = AverageMeter()
    top1 = AverageMeter()
    
    device = next(model.parameters()).device

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            # compute output
            output = model(data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1[0].item(), data.size(0))
    
    print("Epoch-[{:03d}]: Test loss: {:.2f}, acc: {:.2f}.".format(cepoch, losses.avg, top1.avg,))
    
    return top1.avg