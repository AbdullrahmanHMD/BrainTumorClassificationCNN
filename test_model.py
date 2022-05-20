from data.bt_dataset import BrainTumorDataset
from model import BrainTumorCNN
from torch.utils.data import DataLoader
import torch
import os
import cv2

IMAGE_SIZE = 512

# --- Defining the Dataset object ------------------------
dataset_path = os.path.join(os.getcwd(), 'data')
dataset = BrainTumorDataset(dataset_path=dataset_path)
train_loader = DataLoader(dataset, batch_size=1)
# --- Creating the model ---------------------------------

datum = dataset[0][0]

print(datum.shape)
model = BrainTumorCNN()

for x, y, y_txt in train_loader:
    print(model(x).shape)
    break

# print(datum)
# print(model(datum))
# print(datum.shape)