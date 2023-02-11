## Library imports
import os
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import FashionMNIST
from torchsummary import summary
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime

## Set torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

## Initialize parameters
from Non_DL_Classifier.config import data_dir, batch_size, num_epochs, input_size
from Non_DL_Classifier.config import num_classes, learning_rate, momentum

def define_transforms():
    """
    Define transformations for training, validation, and test data.
    For training data we will do resize to 224 * 224, randomized horizontal flipping, rotation, 
    lighting effects, and normalization. For test set we will do only center cropping to get 
    to 224 * 224 and normalization
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(257),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=(0.0,1.5), contrast=(1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    return data_transforms

def create_datasets(data_dir):
    """
    Create datasets for training, validation, and test

    Args:
        data_dir (str): path to data directory

    Returns:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        test_dataset (torchvision.datasets.ImageFolder): test dataset
    """
    ## Define transformations for training, validation, and test data
    data_transforms = define_transforms()

    ## Create Datasets for training and validation sets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                              data_transforms['train'])
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                              data_transforms['val'])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), data_transforms['test'])

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(train_dataset, val_dataset, batch_size=batch_size):
    """
    Create dataloaders for training and validation sets

    Args:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        batch_size (int): batch size

    Returns:
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        dataset_sizes (dict): dictionary of sizes of training and validation sets
        class_names (list): list of class names
        num_classes (int): number of classes
    """
    # Create DataLoaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

    # Set up dict for dataloaders
    dataloaders = {'train':train_loader,'val':val_loader}

    # Store size of training and validation sets
    dataset_sizes = {'train':len(train_dataset),'val':len(val_dataset)}

    # Get class names associated with labels
    class_names = train_dataset.classes
    num_classes = len(class_names)

    return dataloaders, dataset_sizes, class_names, num_classes

class SVM_Loss(nn.modules.Module):
    """
    SVM Loss function
    """    
    def __init__(self):
        """
        Initialize the SVM Loss function
        """
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels):
        """
        Forward pass of the SVM Loss function
        """
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size

def train_and_test_model():
    """
    Driving function for the script. Trains a model on the training set and evaluates it on the validation set.
    Saves the model weights to a file.

    """
    # SVM regression model and Loss
    svm_model = nn.Linear(input_size,num_classes)

    # Loss and optimizer
    svm_loss_criteria = SVM_Loss()
    svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)

    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(data_dir)
    dataloaders, dataset_sizes, class_names, num_classes = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)


    total_step = len(dataloaders["train"])
    for epoch in range(num_epochs):
        avg_loss_epoch = 0
        batch_loss = 0
        total_batches = 0
        for i, (images, labels) in enumerate(dataloaders["train"]):
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, input_size)                      
            
            # Forward pass        
            outputs = svm_model(images)           
            loss_svm = svm_loss_criteria(outputs, labels)    
            
            # Backward and optimize
            svm_optimizer.zero_grad()
            loss_svm.backward()
            svm_optimizer.step()    
            total_batches += 1     
            batch_loss += loss_svm.item()

        # Print loss every few iterations
        avg_loss_epoch = batch_loss/total_batches
        print ('Epoch [{}/{}], Averge Loss:for epoch {}: {:.4f}]' 
                    .format(epoch+1, num_epochs, epoch+1, avg_loss_epoch ))

    # Test the SVM Model
    correct = 0.
    total = 0.
    for images, labels in dataloaders["val"]:

        # Reshape images
        images = images.reshape(-1, input_size)
        
        # Forward pass
        outputs = svm_model(images) 
        
        # Get predictions
        predicted = torch.argmax(outputs, axis=1)

        # Calculate accuracy
        total += labels.size(0) 
        correct += (predicted == labels).sum()    
    
    print('Accuracy of the SVM model on the val images: %f %%' % (100 * (correct.float() / total)))


if __name__ == '__main__':
    train_and_test_model()
