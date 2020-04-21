import shutil
import time

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.models as models

from utils.meters import AverageMeter, ProgressMeter

'''
Collection of methods to apply to CNNs/models
'''


def train_model(model, train_dataset, val_loader, start_epoch, epochs, optimizer, criterion, filename, model_name, args, acc=0.0):
    """
        Wrapper method for training a model.
        Arguments: 
            model - PyTorch Model: the model to train.
            train_dataset - torch.utils.data.DataFolder: dataset image folder directory with augmentations to apply.
            val_loader -  torch.utils.data.DataLoader: validation dataset wrapped in a dataloader.
            start_epoch - int: initial epoch to start/resume training from.
            epochs - int: end epoch 
            optimizer - torch.optim: optimizer to use
            criterion - torch.nn: Loss Function to use.
            filename - str: name of file to wrote training data to. 
            model_name - str: name of file to write the model to. 
            args - dict: arguments passed. 
            acc - float: best accuracy of model seen so far.
    """
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    best = acc
    optim = optimizer
    for epoch in range(start_epoch, epochs):
        
        # adjust learning rate
        if args.decay:
            optim = adjust_learning_rate(optim, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optim, epoch, args)

        # evaluate on validation set
        acc = validate(val_loader, model, criterion, args, filename, epoch)

        if (acc > best):
            best = acc
            state = model.state_dict()
            save(state,model_name)


def train(train_loader, model, criterion, optimizer, epoch, args):
    """
        Train model on train_loader data.

        ### Arguments:
            train_loader -  torch.utils.data.DataLoader: training dataset wrapped in a dataloader.
            model - PyTorch Model: the model to train.
            criterion - torch.nn: Loss Function to use.
            optimizer - torch.optim: optimizer to use
            epoch - int: current training epoch.
            args - dict: arguments passed. 
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, filename, epoch=-1):
    """
        Validate model accuracy on val_loader data.

        ### Arguments:
            val_loader -  torch.utils.data.DataLoader: validation dataset wrapped in a dataloader.
            model - PyTorch Model: the model to validate.
            criterion - torch.nn: Loss Function to use.
            args - dict: arguments passed. 
            filename - str: file to write results to.
        ### Return:
            float- top1 average accuracy.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        # Append results to csv
        results = open(filename, 'a')
        result = '{},{},{}\n'.format(epoch, top1.avg, top5.avg)
        results.write(result)
        results.close()

    return top1.avg



def save(state, model_name, filename='checkpoint.pth.tar'):
    """
        save state to file directory 
    """
    torch.save(state, filename)
    shutil.copyfile(filename, f'{model_name}.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every args.decay epochs"""
    lr = args.lr * (0.1 ** (epoch // args.decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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

def test(test_dataset, model, mapping, args, filename):
    """
        Produces a csv on predictions of the test dataset
    """
    batch_time = AverageMeter('Time', ':6.3f')

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    progress = ProgressMeter(
        len(test_loader),
        [batch_time],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    print (mapping)

    idx_to_class = dict()

    for k, v in mapping.items():
        idx_to_class[v] = k 

    # Append results to csv
    results = open(filename, 'a')

    image = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

            # write classifications
            predictions = pred.cpu().detach().numpy()[0]
            result = ""
            for prediction in predictions:
                test = 'test_{}.JPEG'.format(image)
                classification = idx_to_class[prediction]
                result += '{} {}\n'.format(test, classification)
                image+=1
            results.write(result)
    
    results.close()

def get_model(args, initial=False):
    '''
        Return the model to use.

        ### Arguments:
            args: dict.
            initial: bool - first time the method is being called this run.
        
        ### Returns:
            Torch model.
    '''
    
    if args.own != None:
        print("=> using your pre-trained model '{}'".format(args.arch))
    elif args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](pretrained=args.pretrained)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    if args.own != None and initial:
        if args.loadfinetuned:
            if args.arch.startswith('vgg'):
                number_of_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(number_of_features, args.loadfinetuned)     

                model.classifier[6] = model.classifier[6].cuda()
            else:
                number_of_features = model.module.fc.in_features
                model.module.fc = nn.Linear(number_of_features, args.loadfinetuned)     

                model.module.fc = model.module.fc.cuda()

        model.load_state_dict(torch.load(args.own))
    
    return model
