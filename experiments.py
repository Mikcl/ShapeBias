import os
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformations.custom_transforms import DOG, Gabor

'''
Defining specific experiments
'''    

class Experiment(object):
    '''
        Maintain experiment type
    '''
    def __init__(self, args):
        self.arch = args.arch
        self.finetune = args.finetune
        self.experiment_dataset = args.dataset
        self.concat = args.concat
        self.same = args.same
        self.DOG = args.DOG
        self.DOG_options = args.options
        self.gabor = args.gabor
        self.scales = args.scales
        self.orientations = args.orientations
        self.own = args.own
        self.directory = args.data

        self.name = self.get_name()
        
    def get_name(self):
        '''
            Given type of experiment, return information about experiment as string.
        '''
        f = 'F' if self.finetune else ''

        if self.own and self.finetune:
            return f + str(self.own)

        if self.own:
            return str(own)

        concat_or_same = 'original'
        if self.concat:
            concat_or_same = 'concat'
        elif self.same:
            concat_or_same = 'same'

        transform = 'None'
        if self.DOG:
            transform = 'DOG:'
            if self.DOG_options:
                transform += f'({str(self.DOG_options)})'
        elif self.gabor:
            transform = 'gabor:'
            if self.scales:
                transform += f'v({str(self.scales)})'
            if self.orientations:
                transform += f'u({str(self.orientations)})'


        name = f + str(self.experiment_dataset) + '-' + str(self.arch) + '-' \
            + concat_or_same + '-' + transform
        
        return name    

    def get_transformation_type(self):
        '''
            Get custom transformation to apply, with hyperparameters passed. 
            
            ### Returns:
                Transform object
        '''
        if self.DOG:
            if not self.DOG_options:
                return DOG()
            elif len(self.DOG_options) == 1:
                return DOG(sigma=float(self.DOG_options[0]))
            else:
                return DOG(sigma=float(self.DOG_options[0]), k=float(self.DOG_options[1]))
        if self.gabor:
            if not self.scales and not self.orientations:
                return Gabor()
            if self.scales and not self.orientations:
                return Gabor(scales=[float(s) for s in self.scales])
            if not self.scales and self.orientations:
                return Gabor(orientations=[float(u) for u in self.orientations])
            else:
                return Gabor(scales=[float(s) for s in self.scales] , orientations=[float(u) for u in self.orientations])

        return None

    def get_transformation_set(self):
        '''
            Gets the transformations to apply for the experiment training, additional and validation set

            ### Returns:
                train, additional, validation - transformations to apply respectively 
        '''
        # defined normalization values from image-net.
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        original_transformation = [transforms.ToTensor(), normalize]
        custom_transform = self.get_transformation_type()
        additional_transformation = [custom_transform,transforms.ToTensor()]

        if self.same and custom_transform:
            # Only train and test on a custom transformation
            return  transforms.Compose(additional_transformation), None, transforms.Compose(additional_transformation)
        if custom_transform:
            # Concat with custom transform
            return  transforms.Compose(original_transformation), transforms.Compose(additional_transformation), transforms.Compose(original_transformation)
        else: 
            # No additional data
            return  transforms.Compose(original_transformation), None, transforms.Compose(original_transformation)

    # Datasets for differnet experiments
    def get_data_set(self, transform, additional_transform, validation_transform):
        '''
            Gets the training and test dataset for an experiment, applying relevant transformation and concatenations

            ### Arguments:
                transform: list('Transform') - list of Transform objects - applied to training
                additional_transform: list('Transform') - list of Transform objects - applied to training set
                validation_transform: list('Transform') - list of Transform objects to be applied to validation set
            
            ### Returns:
                'torchvision.dataset', 'torchvision.dataset' - pair corresponding to training and validation set
        '''
        if self.experiment_dataset == None:
            # Custom Dataset - with directory/train and directory/val sub folders
                traindir = os.path.join(self.directory, 'train')
                valdir = os.path.join(self.directory, 'val')
        
                train_dataset = self.define_dataset(traindir, transform)
                test_dataset = self.define_dataset(valdir, validation_transform)

                if self.concat:
                    transformed_train_dataset = self.define_dataset(traindir, additional_transform)
                    train_dataset = train_dataset + transformed_train_dataset 

        elif self.experiment_dataset == 0:
                train_dataset = datasets.CIFAR10(root=self.directory, train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR10(root=self.directory, train=False, download=True, transform=validation_transform)
                
                if self.concat:
                    transformed_train_dataset = datasets.CIFAR10(root=self.directory, train=True, download=True, transform=additional_transform)
                    train_dataset = train_dataset + transformed_train_dataset

        elif self.experiment_dataset == 1:
                train_dataset = datasets.CIFAR100(root=self.directory, train=True, download=True, transform=transform)
                test_dataset = datasets.CIFAR100(root=self.directory, train=False, download=True, transform=validation_transform)
                
                if self.concat:
                    transformed_train_dataset = datasets.CIFAR100(root=self.directory, train=True, download=True, transform=additional_transform)
                    train_dataset = train_dataset + transformed_train_dataset

        return train_dataset, test_dataset

    def define_dataset(self, directory, augmentations):
        '''
            Defines a custom dataset. 

            ### Arguments:
                directory: str - directory where the dataset is located.
                augmentations: 'transforms.Compose'  - a list of Transforms to apply.
            
            ### Returns:
                'torchvision.dataset' - dataset image folder with augmentations to apply
        '''
        dataset = datasets.ImageFolder(directory,augmentations)
        
        return dataset
