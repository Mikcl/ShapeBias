
def set_args(parser):
    '''
        separate method to add command line arguments

        ### Arguments:
            parser: 'argParse.ArgumentParser'
        ### Returns:
             'argParse.ArgumentParser'
    '''
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=74, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--own', default=None, type=str, metavar='PATH',
                        help='path to your own given model state dict (note arch of this model must match -a)')
    parser.add_argument('-f', '--finetune', dest='finetune', action='store_true',
                        help='fine-tune model fc layer on training set')
    parser.add_argument('-l', '--loadfinetuned', default=None, type=int, metavar='LF',
                        help='use fintuned own model with defined number of output classes')
    parser.add_argument('-c', '--channels', default=3, type=int,
                        help='number of channels (default: 3)')
    parser.add_argument('--decay', default=0, type=int, metavar='D',
                        help='decay learning rate every D epochs by factor of 10')
    parser.add_argument('-d', '--dataset', default=None, type=int,
                        help='which dataset to train on, default- None (define custom path to directory), 0 - CIFAR10, 1 - CIFAR100')
    parser.add_argument('--data', metavar='DATA', default='./',
                        help='path to custom dataset and where output folder is')
    parser.add_argument('--concat', dest='concat', action='store_true',
                        help='concat transformed data with original data')
    parser.add_argument('--same', dest='same', action='store_true',
                        help='train and validated on same (type of transformation) dataset, primarily used from custom transformations') 
    parser.add_argument('--DOG', dest='DOG', action='store_true',
                        help='DOG transformation, use --options for non default hyper parameters, [sigma k]')
    parser.add_argument('-o','--options', nargs='+', default=None, metavar='O',
                        help='options (list) for the transformation, pass as: -o s k')
    parser.add_argument('--gabor', dest='gabor', action='store_true',
                        help='gabor 2D CWT')
    parser.add_argument('-s','--scales', nargs='+', default=None, metavar='S',
                        help='scales (list) for the 2D Gabor Wavelet: -s 2 2.5')
    parser.add_argument('-u','--orientations', nargs='+', default=None, metavar='U',
                        help='orientations (list) for the 2D Gabor Wavelet: -u 1 2 3')
    parser.add_argument('--savecsv', dest='savecsv', action='store_true',
                        help='save the csv to google drive [only in collab]')
    parser.add_argument('--savemodel', dest='savemodel', action='store_true',
                        help='save the model to google drive [only in collab]')

    return parser