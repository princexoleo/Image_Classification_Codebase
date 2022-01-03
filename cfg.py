"""
Created by Mazharul Islam Leon
Created on Wed Dec 15 2021

This is main config file of the image classifier

"""

import os
from pathlib import Path


# Home directory of the projects
HOME = Path.home()
#HOME = os.path.expanduser("C:\Users\iceba\workspace\ML\wbc\codebase\Image_Classification_Codebase")


# Number of Class in the dataset
NUM_CLASSES = 10

# By default the bactch size is set to 32
BATCH_SIZE = 32

# By default the inpur image size is set to 224
INPUT_SIZE = 224

# By default the epochs is set to 10
EPOCHS = 10

# By default the MAX EPOCH is set to 100
MAX_EPOCH = 100

# By default the number of GPU is set to 1
GPUS = 0

# By default the Resume Epoch is set to 0
RESUME_EPOCH = 0

# By default the learning rate is set to 0.001
LEARNING_RATE = 0.001

# By default the weight decay is set to 0.0001
WEIGHT_DECAY = 0.0001

# By default the momentum is set to 0.9
MOMENTUM = 0.9

# By default the learning rate (LR) is set to 0.001  
LR = 0.001

# By default the model name is set to "ResNet50"
# MODEL_NAME = "resnet50"
model_name = 'resnet50'

# 
from models import Resnet50, Resnet101, Resnext101_32x8d,Resnext101_32x16d, Densenet121, Densenet169, Mobilenetv2, Efficientnet, Resnext101_32x32d, Resnext101_32x48d


# Creating a dictonary to store the model_name and corresponding to their actual name   
MODEL_NAMES = {
     'resnext101_32x8d': Resnext101_32x8d,
    'resnext101_32x16d': Resnext101_32x16d,
    'resnext101_32x48d': Resnext101_32x48d,
    'resnext101_32x32d': Resnext101_32x32d,
    'resnet50': Resnet50,
    'resnet101': Resnet101,
    'densenet121': Densenet121,
    'densenet169': Densenet169,
    'moblienetv2': Mobilenetv2,
    'efficientnet-b7': Efficientnet,
    'efficientnet-b8': Efficientnet
    }

# By default the base_dir is set to "./data"
#BASE = os.path.join(HOME, "workspace\ML\wbc\codebase\Image_Classification_Codebase\data")
BASE =  Path(".")

# By default the model weight saving path is set to BASE + "weights"
# MODEL_SAVE_PATH = BASE + "weights/"
# SAVE_FOLDER = BASE + 'weights\\'
SAVE_FOLDER = BASE / "weights"

# Training label path
# TRAIN_LABEL_DIR =BASE + '\\train.txt'     
# VAL_LABEL_DIR = BASE + '\\val.txt'
# TEST_LABEL_DIR = BASE + '\\test.txt'

TRAIN_LABEL_DIR = BASE / "data" / "train.txt"   
VAL_LABEL_DIR = BASE / "data" / "val.txt"      
TEST_LABEL_DIR = BASE / "data" / "test.txt"   


# By default the trained model file name is set to "model.pth"
# TRAINED_MODEL = BASE + "weights/resnext101_32x32d/epoch_40.pth"
TRAINED_MODEL = BASE / "weights" / "resnet50" / "epoch_40.pth"
# by default train dir is set to "./data/train"
# TRAIN_LABEL_DIR =  BASE + "\\train.txt"
