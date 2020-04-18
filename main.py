import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

import experiments
from experiments import Experiment
from utils.meters import AverageMeter, ProgressMeter
import utils.modelling as modelling
import extend_parser

'''
Main driver code to run experiments
'''

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Command line arguments
parser = argparse.ArgumentParser(description='Shape Bias Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser = extend_parser.set_args(parser)

def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    # Simply call main_worker function
    main_worker(ngpus_per_node, args)

def main_worker(ngpus_per_node, args):

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    experiment = Experiment(args)

    # Create file and store in args.data directory...
    filename = f'{experiment.name}.csv'
    print ("Creating {} to store accuracy results".format(filename))
    results = open(filename, 'w+')
    results.write("Epoch,Top1,Top5\n")
    results.close()


    # create model
    model = modelling.get_model(args, True)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Get Transformations.
    transform, additional_transform, validation_transform = experiment.get_transformation_set()

    # Get Data
    train_dataset, val_dataset = experiment.get_data_set(transform, additional_transform, validation_transform)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Evaluate loaded model.
    if args.evaluate:
        modelling.validate(val_loader, model, criterion, args, filename)
        return 

    # Train your own model
    if args.own == None:
        modelling.train_model(model, train_dataset, val_loader, args.start_epoch, args.epochs, optimizer, criterion, filename, args)
    
    # Fine Tune the model.
    if args.finetune:
        
        if args.arch.startswith('vgg'):
            number_of_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(number_of_features, len(train_dataset.classes))     

            model.classifier[6] = model.classifier[6].cuda()
        else:
            number_of_features = model.module.fc.in_features
            model.module.fc = nn.Linear(number_of_features, len(train_dataset.classes))     

            model.module.fc = model.module.fc.cuda()

        # only fc layer is are being updated. 
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                            args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        
        modelling.train_model(model, train_dataset, val_loader, args.start_epoch, args.epochs, optimizer, criterion, filename, args)
        

if __name__ == '__main__':
    main()