import torch
import datautil
import torch.autograd
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import torch.utils.data
import sys,os,os.path
from ensemble import Ensemble

parser = argparse.ArgumentParser(description='scene class')
parser.add_argument('--data', metavar='PATH',
                    help='path to dataset')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--img_size',default=320,type=int,metavar='N',help='size of image')
parser.add_argument('--test_model',default='checkpoint/resnet18_bestof90epoch.pth.tar')
parser.add_argument('-a','--arch',default='ensemble',type=str)
parser.add_argument('--num_classes',default=10,type=int)
parser.add_argument('-k',default=3,type=int)

def test_labeled(test_loader,model):
    top1 = AverageMeter()
    top3 = AverageMeter()

    model.eval()

    for i,(input,label,names) in enumerate(test_loader):
        label = label.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(label, volatile=True)

        output = model(input_var)
        pred = torch.max(output.data, 1)  # batchsize*20->batchsize

        prec1,prec3 = accuracy(output.data,label,topk=(1,3))
        top1.update(prec1[0],input.size(0))
        top3.update(prec3[0],input.size(0))
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   i, len(test_loader),
                   top1=top1, top3=top3))
    print(' * Prec@1 {top1.avg:.3f} prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))


def test(test_loader, model):
    model.eval()
    ret=[]
    for i,input in enumerate(test_loader):
        input_val = torch.autograd.Variable(input,volatile=True)

        output = model(input_val)
        #pred = torch.max(output.data,1)[1]
        pred = output.topk(args.k,1)[1] #(N,3) index
        ret.extend([i for i in pred])
        printoneline('%.2f' % (i/len(test_loader)*100),'%')
    return ret


def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()


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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    print(args)

    print("=> creating model '{}'".format(args.arch))
    model = Ensemble()
    model = torch.nn.DataParallel(model).cuda()
    print(model)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])
    test_data = datautil.SceneDataset(args.data,img_transform=
                                            transforms.Compose([
                                                transforms.Resize((args.img_size,args.img_size)),
                                                transforms.ToTensor(),
                                                normalize]))
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    checkpoint = torch.load(args.test_model)
    model.load_state_dict(checkpoint['state_dict'])
    #model.load_state_dict(checkpoint)
    if os.path.isdir(args.data):
        ret = test(test_loader,model)
        imgs = [i[:-4] for i in os.listdir(args.data)]
        with open('result3_.csv', 'w') as f:
            '''
            f.write(','.join(['FILE_ID','CATEGORY_ID'])+'\n')
            f.write('\n'.join([','.join([str(a),str(b)]) for a,b in zip(imgs,ret)]))
            '''
            #FILE_ID,CATEGORY_ID0,CATEGORY_ID1,CATEGORY_ID2
            f.write(','.join(['FILE_ID','CATEGORY_ID0','CATEGORY_ID1','CATEGORY_ID2'])+'\n')
            f.write('\n'.join([','.join([str(a)]+[str(int(i)) for i in b]) for a,b in zip(imgs,ret)]))
    else:
        test_labeled(test_loader,model)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
