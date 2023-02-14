## Description: This file contains the configuration parameters for the training script

# Model filename
model_dir = "models/resnet50classifier.pkl"
pretrained_model_to_use = "resnet18"
freeze_pretrained_model = True

# Data directory
# Current directory has just a sub sample of images, replace with the full dataset
data_dir = 'data/toys_data'

# Batch size
batch_size = 8

# Epochs
num_epochs = 10

# Data split percentages
train_percentage = 0.5
val_percentage = 0.25
test_percentage = 0.25

# Number of workers
num_workers = 1