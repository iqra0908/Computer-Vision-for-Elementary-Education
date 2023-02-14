# library imports
import torch
import torchvision
import torchvision.transforms as transforms
import time

def define_transforms():
    """
    Define transformations for training, validation, and test data.
    For training data we will do resize to 224 * 224, randomized horizontal flipping, rotation, lighting effects, and normalization. 
    For test and val set we will do only center cropping to get to 224 * 224 and normalization
    """

    data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    return data_transforms


def create_datasets(data_dir, train_percentage, val_percentage):
    """
    Create datasets for training, validation, and test

    Args:
        data_dir (str): path to data directory
        train_percentage (float): percentage of data to use for training
        val_percentage (float): percentage of data to use for validation

    Returns:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        test_dataset (torchvision.datasets.ImageFolder): test dataset
        class_names (list): list of class names
        num_classes (int): number of classes
    """
    ## Define transformations for training, validation, and test data
    data_transforms = define_transforms()

    ## Create Datasets for training, testing and validation sets
    image_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=data_transforms)
    train_size = int(train_percentage * len(image_dataset))
    val_size = int(val_percentage * len(image_dataset))
    test_size = len(image_dataset) - train_size - val_size

    ## Split the dataset into training, validation and test sets
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, val_size, test_size])

    ## get class names associated with labels
    class_names = image_dataset.classes
    num_classes = len(class_names)

    return train_dataset, val_dataset, test_dataset, class_names, num_classes


def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size, num_workers=2):
    """
    Create dataloaders for training and validation and testing sets

    Args:
        train_dataset (torchvision.datasets.ImageFolder): training dataset
        val_dataset (torchvision.datasets.ImageFolder): validation dataset
        test_dataset (torchvision.datasets.ImageFolder): test dataset
        batch_size (int): batch size
        num_workers (int): number of workers to use for dataloader

    Returns:
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        dataset_sizes (dict): dictionary of sizes of training and validation sets
    """
     
    ## Create DataLoaders for training, testing and validation sets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                            shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers)

    ## Set up dict for dataloaders
    dataloaders = {'train':train_loader, 'val':val_loader, 'test': test_loader}

    ## Store size of training and validation sets
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset)}

    return dataloaders, dataset_sizes


def train_model(model, input_size, criterion, optimizer, dataloaders, batch_size, device="cpu", num_epochs=1):
    """
    Train the model using transfer learning
    Args:
        model (torchvision.models): model to train
        input_size (int): input size of the model
        criterion (torch.nn.modules.loss): loss function
        optimizer (torch.optim): optimizer
        dataloaders (dict): dictionary of dataloaders for training and validation sets
        device (torch.device): device to train on
        num_epochs (int): number of epochs to train for
    Returns:
        model (torchvision.models): trained model
    """
    ## Load the model to GPU if available
    model = model.to(device)

    ## Train the model
    for epoch in range(num_epochs):
        avg_loss_epoch = 0
        batch_loss = 0
        total_batches = 0

        for images, labels in dataloaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            images = images.reshape(-1, input_size)
            optimizer.zero_grad()
            
            ## Forward pass        
            outputs = model(images)           
            loss_svm = criterion(outputs, labels, batch_size)    
            
            ## Backward and optimize
            loss_svm.backward()
            optimizer.step()    
            total_batches += 1     
            batch_loss += loss_svm.item()

        ## Print loss every few iterations
        avg_loss_epoch = batch_loss/total_batches
        print ('Epoch [{}/{}], Averge Loss:for epoch {}: {:.4f}]'.format(epoch+1, num_epochs, epoch+1, avg_loss_epoch))
    return model

def test_model(model, test_dataloader, device, input_size):
    """
    Test the trained model performance on test dataset
    Args:
        model (torchvision.models): model to train
        test_dataloader (torch.utils.data.DataLoader): test dataloader
    Returns:
        model (torchvision.models): trained model
    """
    ## Load the model to GPU if available
    model = model.to(device)

    ## Set model to evaluate mode
    model.eval()

    correct = 0.
    total = 0.

    ## Iterate through test dataset
    for images, labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)
        ## Reshape images
        images = images.reshape(-1, input_size)
        
        ## Forward pass
        outputs = model(images) 
        
        ## Get predictions
        predicted = torch.argmax(outputs, axis=1)

        ## Calculate accuracy
        total += labels.size(0) 
        correct += (predicted == labels).sum()    

    print('Accuracy of the SVM model on the val images: %f %%' % (100 * (correct.float() / total)))