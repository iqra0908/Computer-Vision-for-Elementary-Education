## Description: This file contains the configuration parameters for the training script
import datetime

# Data directory
data_dir = 'data/toys_data' # for google drive: '/content/gdrive/MyDrive/AIPI540/'

# Parameters for training
batch_size = 8
num_epochs = 35 
input_size = 150528  #(dimension of image 224*224*3)
num_classes = 8  
learning_rate = 0.0001 ## step size used by SGD 
momentum = 0.1 ## Momentum is a moving average of our gradients (helps to keep direction)