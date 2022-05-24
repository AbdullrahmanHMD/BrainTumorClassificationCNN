from data.bt_dataset import BrainTumorDataset
from model2 import BrainTumorCNN
from torch.utils.data import DataLoader
import torchvision
import torch
import os
import cv2
from data.preprocessing import *

IMAGE_SIZE = 256

# --- Defining the Dataset object ------------------------
dataset_path = os.path.join(os.getcwd(), 'data')

preprocessing_ops = [resize([256, 256]), sharpen()]

dataset = BrainTumorDataset(dataset_path=dataset_path, preprocessing=preprocessing_ops)
train_loader = DataLoader(dataset, batch_size=1)

# --- Creating the model ---------------------------------
 
datum = dataset[0][0]

# dataset.plot_data_distribution()

curr_images = []

model = BrainTumorCNN()

# i = 0
for x, y, y_txt in train_loader:
    
    yhat = model(x)

    break
