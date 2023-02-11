## Library imports
import os
import copy
import time
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
from ResnetTraining.config import model_filename, data_dir, batch_size, num_epochs


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

def train_model(model, criterion, optimizer, dataloaders, scheduler, device, num_epochs=num_epochs):
    """
    Train the model using transfer learning

    Args:
        model (torchvision.models): model to train
        criterion (torch.nn.modules.loss): loss function
        optimizer (torch.optim): optimizer
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        scheduler (torch.optim.lr_scheduler): learning rate scheduler
        device (torch.device): device to train on
        num_epochs (int): number of epochs to train for

    Returns:
        model (torchvision.models): trained model
    """
    # Send model to GPU if available
    model = model.to(device) 
    since = time.time()

    # Initialize best model weights and accuracy
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Loop over epochs
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Get the input images and labels, and send to GPU if available
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += loss.item() * inputs.size(0)
                # Track number of correct predictions
                running_corrects += torch.sum(preds == labels.data)

            # Step along learning rate scheduler when in train
            if phase == 'train':
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # If model performs better on val set, save weights as the best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './model/best_model.pt')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:3f}'.format(best_acc))

    # Load the weights from best model
    model.load_state_dict(best_model_wts)

    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('[INFO] Device is {device}')


def run_script():
    """
    Driving function for the script. Trains a model on the training set and evaluates it on the validation set.
    Saves the model weights to a file.

    """
    # Instantiate pre-trained resnet
    net = torchvision.models.resnet50(pretrained=True)
    
    # Shut off autograd for all layers to freeze model so the layer weights are not trained
    for param in net.parameters():
        param.requires_grad = False

    # Get the number of inputs to final Linear layer
    num_ftrs = net.fc.in_features
    # Replace final Linear layer with a new Linear with the same number of inputs but just num_classes outputs,

    net.fc = nn.Linear(num_ftrs, num_classes)

    # Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Learning rate scheduler - decay LR by a factor of 0.1 every 7 epochs
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset = create_datasets(data_dir)
    dataloaders, dataset_sizes, class_names, num_classes = create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)

    # Train the model
    net = train_model(net, criterion, optimizer, dataloaders, lr_scheduler, device, num_epochs=num_epochs)

if __name__ == '__main__':
    run_script()

