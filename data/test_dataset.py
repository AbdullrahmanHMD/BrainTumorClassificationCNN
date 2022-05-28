from bt_dataset import *
import cv2
from preprocessing import *

import os

dataset_path = os.getcwd()


preprocessing_ops = [denoise, resize([IMAGE_SIZE, IMAGE_SIZE]), sharpen()]
dataset = BrainTumorDataset(dataset_path=dataset_path, preprocessing=preprocessing_ops)
