from data.bt_dataset import BrainTumorDataset
from model import BrainTumorCNN
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

print(datum.shape)
model = BrainTumorCNN()
i = 0
for x, y, y_txt in train_loader:
    
    print(x.shape)
    
    # try:
    # except Exception:
        # curr_images.append(list(x.shape[2:]))
        # print(dataset.data_paths[i])
        # print(x.shape)
    # i += 1

# print(curr_images)

# with open('image_sizes.txt', 'w') as file:
    # for size in curr_images:
        # file.write(str(size[0]) + " " + str(size[1]))
        # file.write('\n')
# print(model(datum))
# print(datum.shape)