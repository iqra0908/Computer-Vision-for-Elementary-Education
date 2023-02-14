## library imports
import torch
import torch.nn as nn
import torch.optim as optim
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
from helper import create_datasets, create_dataloaders, train_model, test_model
from config import model_dir, input_size, num_classes, learning_rate, momentum
from config import data_dir, train_percentage, val_percentage, test_percentage, num_workers
from config import batch_size, num_epochs

## Torch parameters being used
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
logging.info(f'Torch Version: {TORCH_VERSION}, Cuda Version: {CUDA_VERSION}')

class SVM_Loss(torch.nn.modules.Module):
    """
    SVM Loss function
    """    
    def __init__(self):
        """
        Initialize the SVM Loss function
        """
        super(SVM_Loss,self).__init__()

    def forward(self, outputs, labels, batch_size):
        """
        Forward pass of the SVM Loss function
        """
        return torch.sum(torch.clamp(1 - outputs.t()*labels, min=0))/batch_size


def main():
    """
    Driving function for the script. Trains a model on the training set while evaluating it on the
    validation set. Saves the best model 
    """

    ## Create datasets and dataloaders
    train_dataset, val_dataset, test_dataset, class_names, num_classes = create_datasets(data_dir, train_percentage, val_percentage)
    logging.info('Train, Validation and Test Datasets Created Successfully.')
    
    dataloaders, dataset_sizes = create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers)
    logging.info('Train, Validation and Test Dataloaders Created Successfully.')

    ## Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Device {device} Being Used.')

    ## SVM regression model and Loss
    svm_model = nn.Linear(input_size,num_classes)

    ## Loss and optimizer
    svm_loss_criteria = SVM_Loss()
    svm_optimizer = torch.optim.SGD(svm_model.parameters(), lr=learning_rate, momentum=momentum)
    total_step = len(dataloaders["train"])

    ## Train model
    model = train_model(svm_model, input_size, svm_loss_criteria, svm_optimizer, dataloaders, batch_size, device, num_epochs)

    ## Save model
    torch.save(model, model_dir)
    logging.info('Model Saved Successfully.')
    
    
    ## Test model
    model = torch.load(model_dir)
    #test_model(model, dataloaders["test"], device, input_size)

if __name__ == "__main__":
    main()