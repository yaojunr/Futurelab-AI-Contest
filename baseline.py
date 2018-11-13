import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboard_logger import configure, log_value

from ensemble import Ensemble

parser = argparse.ArgumentParser(description='Scene Classification')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='ensemble',
                    help='model architecture: ' +
                        ' (default: mobilenetV2)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.99, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '-wd', default=1e-3, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--num_classes',default=10, type=int, help='num of class in the model')
parser.add_argument('--val',type=str,metavar='PATH',help='validate data list')
parser.add_argument('--img_size',default=320,type=int,metavar='N',help='size of image')
parser.add_argument('--interval',default =1,type=int,metavar='N',help='frequency of save checkpoint')
parser.add_argument('--tensorboard',dest='tensorboard',action='store_true')

best_prec1 = 0
step = 0

def main():
    global best_prec1
    if args.tensorboard:
        configure('log/'+args.arch.lower() + '_bs' + str(args.batch_size) + '_ep' + str(args.epochs) + '_loglr' + str(args.lr) +
                '_size' + str(args.img_size)+ '_wd' + str(args.weight_decay))
    print(args)
    # create model
    print("=> creating model '{}'".format(args.arch))
    
    #if args.arch.lower().startswith('resnet'):
    #    model.avgpool = nn.AvgPool2d(args.img_size // 32, 1)
    #model.fc = nn.Linear(model.fc.in_features, args.num_classes)

    # default parameter n_class=1000, input_size=224, width_mult=1.
    model = Ensemble()
    if not args.resume:
        model.MobileNetV2.load_state_dict(torch.load('mobilenet_pretrained.pth'))
        model.NASNetAMobile.load_state_dict(torch.load('nasnet_pretrained.pth'))

    model = torch.nn.DataParallel(model).cuda()
    print(model)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    import datautil
    '''
    norm_dict = {
    #320:transforms.Normalize(mean=[0.4333,0.4429,0.4313],std=[ 1.,  1.,  1.]),
    320:transforms.Normalize(mean=[0.4333,0.4429,0.4313],std=[0.2295,  0.2385,  0.2479]),
    0:transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    256:transforms.Normalize(mean=[0.4333,0.4429,0.4313],std=[ 0.2295,  0.2385,  0.2479]),
    224:transforms.Normalize(mean=[0.4333,0.4429,0.4313],std=[ 0.2295,  0.2385,  0.2479]),
    }
    norm_default = norm_dict[0]
    normalize = norm_dict[args.img_size]

    currrent 
    tensor([[ 0.4828,  0.4693,  0.4602]], device='cuda:0')
    tensor([[ 45.3332,  41.1241,  45.7719]], device='cuda:0')

    '''
    #normalize = transforms.Normalize(mean=[0.48280172,0.46929353,0.46019437],std=[0.25859008,0.28414325,0.288328])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    train_data = datautil.SceneDataset(args.data,img_transform=
                                             transforms.Compose([
                                             transforms.RandomResizedCrop(args.img_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             normalize]))


    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
    if args.val:
        val_data = datautil.SceneDataset(args.val, img_transform=
                                        transforms.Compose([
                                            #transforms.Scale(256),
                                            transforms.Resize((args.img_size,args.img_size)),
                                            transforms.ToTensor(),
                                            normalize]))
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size//2, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                 momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.RMSprop(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                            weight_decay=args.weight_decay,eps=1)

    if args.evaluate:
        validate(val_loader, model, criterion,0)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        if args.val:
            prec1 = validate(val_loader, model, criterion,epoch)

        # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
        if epoch % args.interval == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
            })



def train(train_loader, model, criterion, optimizer, epoch):
    global step
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    #std = torch.zeros((3,1,1))
    #mean = torch.tensor([0.4333,0.4429,0.4313]).view(3,1,1)
    for i, (input, target) in enumerate(train_loader):
        #std += ((torch.sum(torch.sum(img,dim=1),dim=1)/(320*320)-mean)**2).view(3,1,1)
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    if args.tensorboard:
        log_value('train_loss',losses.avg,epoch)
        log_value('train_acc1', top1.avg,epoch)

def validate(val_loader, model, criterion,epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    global step

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,),epoch=epoch)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    if args.tensorboard:
            log_value('val_loss',losses.avg,epoch)
            log_value('val_acc1', top1.avg,epoch)

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state,filename='checkpoint.pth.tar'):
    torch.save(state,'new_model.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.98 ** epoch)
    #lr = args.lr * (0.1 ** ((epoch-70))//10)
    print('current lr = {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,),names=None,epoch=''):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    if names is not None:
        with open(os.path.join(model_save_dir,'wrong.txt'),'a') as f:
            for i in range(len(names)):
                if correct[:,i].float().sum()==0:
                    f.write(str(epoch)+':'+names[i]+str(list(pred[:,i].cpu().numpy()))+'\n')
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    main()
