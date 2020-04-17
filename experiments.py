import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformations.custom_transforms import DOG, Gabor


# Datasets for differnet experiments
def get_data_set(experiment, concatenante, transform, additional_transform, validation_transform, directory='./data'):
    '''
        Gets the training and test dataset for an experiment, applying relevant transformation and concatenations

        ### Arguments:
            experiment: int - number pointing to the original dataset experiment, Or None with custom dataset.
            concatenante: bool - true will combine the experiment dataset with the additional_transform-transformed dataset
            transform: list('Transform') - list of Transform objects - applied to training
            additional_transform: list('Transform') - list of Transform objects - applied to training set
            validation_transform: list('Transform') - list of Transform objects to be applied to validation set
            directory: str - path from root to the dataset, supply for custom dataset
        
        ### Returns:
            'torchvision.dataset', 'torchvision.dataset' - pair corresponding to training and validation set
    '''
    if experiment == None:
        # Custom Dataset - with directory/train and directory/val sub folders
            traindir = os.path.join(directory, 'train')
            valdir = os.path.join(directory, 'val')
    
            train_dataset = define_dataset(traindir, transform)
            test_dataset = define_dataset(valdir, transform)

            if concatenante:
                transformed_train_dataset = define_dataset(traindir, additional_transform)
                train_dataset = train_dataset + transformed_train_dataset 

    elif experiment == 0:
            train_dataset = datasets.CIFAR10(root=directory, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(root=directory, train=False, download=True, transform=transform)
            
            if concatenante:
                transformed_train_dataset = datasets.CIFAR10(root=directory, train=True, download=True, transform=additional_transform)
                train_dataset = train_dataset + transformed_train_dataset

    elif experiment == 1:
            train_dataset = datasets.CIFAR100(root=directory, train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR100(root=directory, train=False, download=True, transform=transform)
            
            if concatenante:
                transformed_train_dataset = datasets.CIFAR100(root=directory, train=True, download=True, transform=additional_transform)
                train_dataset = train_dataset + transformed_train_dataset

    return train_dataset, test_dataset

def define_dataset(directory, augmentations):
    '''
        Defines a custom dataset. 

        ### Arguments:
            directory: str - directory where the dataset is located.
            augmentations: list('Transform') - list of augmentations to apply.
        
        ### Returns:
            'torchvision.dataset' - dataset image folder with augmentations to apply
    '''
    dataset = datasets.ImageFolder(
        directory,
        transforms.Compose(augmentations))
    
    return dataset

def get_transformation_type(args):
    '''
        Get custom transformation to apply, with hyperparameters passed. 
        
        ### Arguments:
            args: dict.
        
        ### Returns:
            Transform object
    '''
    if args.gabor:
        if not args.scales and not args.orientations:
            return Gabor()
        if args.scales and not args.orientations:
            return Gabor(scales=[float(s) for s in args.scales])
        if not args.scales and args.orientations:
            return Gabor(orientations=[float(u) for u in args.orientations])
        else:
            return Gabor(scales=[float(s) for s in args.scales] , orientations=[float(u) for u in args.orientations])
    if args.DOG:
        if not args.options:
            return DOG()
        elif len(args.options) == 1:
            return DOG(sigma=float(args.options[0]))
        else:
            return DOG(sigma=float(args.options[0]), k=float(args.options[1]))

    return None

def get_transformation_set(args):
    '''
        ### Arguments:
            args: dict - 
        ### Returns:
            train, additional, validation - transformations to apply respectively 
    '''
    # defined normalization values from image-net.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    original_transformation = [transforms.ToTensor(), normalize]
    custom_transform = get_transformation_type(args)
    additional_transformation = [custom_transform,transforms.ToTensor()]

    if args.same and custom_transform:
        # Only train and test on a custom transformation
        return  transforms.Compose(additional_transformation), None, transforms.Compose(additional_transformation)
    if custom_transform:
        # Concat with custom transform
        return  transforms.Compose(original_transformation), transforms.Compose(additional_transformation), transforms.Compose(original_transformation)
    else: 
        # No additional data
        return  transforms.Compose(original_transformation), None, transforms.Compose(original_transformation)