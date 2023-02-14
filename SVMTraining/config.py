## Description: This file contains the configuration parameters for the training script

# Model filename
model_dir = "models/svm.pt" #path for colab: /content/Computer-Vision-for-Elementary-Education/models/svm.pt

# Data directory
data_dir = 'data/toys_data' #path for colab: /content/Computer-Vision-for-Elementary-Education/data/toys_data

# Parameters for training
batch_size = 8 # Batch size for training (change depending on how much memory you have)
num_epochs = 35 # Number of epochs to train for
input_size = 150528  #(dimension of image 224*224*3)
num_classes = 8 # Number of classes in the dataset
learning_rate = 0.0001 ## step size used by SGD 
momentum = 0.1 # Momentum is a moving average of our gradients (helps to keep direction)

# Data split percentages
train_percentage = 0.5
val_percentage = 0.25
test_percentage = 0.25

# Number of workers
num_workers = 4