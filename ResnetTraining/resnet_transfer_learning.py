## library imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
import json
import random
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = 120000000

## Set random seeds for reproducibility
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

## Logging setup
import logging
logging.basicConfig(level=logging.INFO)

## Local imports
from helper import create_datasets, create_dataloaders, train_model
from config import model_filename, model_dir, pretrained_model_to_use, freeze_pretrained_model
from config import data_dir, train_percentage, val_percentage, test_percentage, num_workers
from config import batch_size, num_epochs


## Torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
logging.info(f'Torch Version: {TORCH_VERSION}, Cuda Version: {CUDA_VERSION}')


## Main function
def run_script():
    """
    Driving function for the script. Trains a model on the training set while evaluating it on
    the validation set. Saves the best model to a file
    """
 
    # create datasets and dataloaders
    train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(data_dir, train_percentage, val_percentage)
    logging.info('Train, Validation and Test Datasets Created Successfully.')
    
    dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)
    logging.info('Train, Validation and Test Dataloaders Created Successfully.')

    ## Instantiate pre-trained resnet
    if pretrained_model_to_use == 'resnet18':
        net = torchvision.models.resnet18(pretrained=True)
    elif pretrained_model_to_use == 'resnet34':
        net = torchvision.models.resnet34(pretrained=True)
    elif pretrained_model_to_use == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
    elif pretrained_model_to_use == 'resnet101':
        net = torchvision.models.resnet101(pretrained=True)
    elif pretrained_model_to_use == 'resnet152':
        net = torchvision.models.resnet152(pretrained=True)
    else:
        print("Invalid model name, exiting...")
    
    logging.info('Model Loaded Successfully.')
    

    ## Freeze model weights for all layers to freeze model so the layer weights are not trained
    if freeze_pretrained_model:
        for param in net.parameters():
            param.requires_grad = False
    
    ## Get the number of inputs to final FC layer
    num_ftrs = net.fc.in_features

    ## Replace existing FC layer with a new FC layer having the same number of inputs and num_classes outputs
    net.fc = nn.Linear(num_ftrs, num_classes)
    
    # ## Show model architecture
    temp, temp_ = next(iter(dataloaders['train']))
    logging.info('\nModel Architecture:')
    print(summary(net,(temp.shape[1:]), batch_size=batch_size, device="cpu"), "\n")

    ## Cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.CrossEntropyLoss()

    ## define optimizer
    if freeze_pretrained_model:
        optimizer = optim.Adam(net.fc.parameters(), lr=0.001)
    else:
        optimizer = optim.Adam(net.parameters(), lr=0.001)

    ## Learning rate scheduler - not using as we used adam optimizer
    lr_scheduler = None

    ## Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device {device} Being Used.')

    ## Train the model
    logging.info('Started Training The Model.\n')
    net = train_model(model=net, model_dir=model_dir, criterion=criterion, optimizer=optimizer, 
                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, scheduler=lr_scheduler, 
                    device=device, num_epochs=num_epochs)

    ## Test the model
    #test_model(model = net, test_dataset = test_dataset, device = device)

if __name__ == '__main__':
    run_script()