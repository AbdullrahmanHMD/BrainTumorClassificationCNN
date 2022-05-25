from bt_dataset import *
import cv2


import os

dataset_path = os.getcwd()

dataset = BrainTumorDataset(dataset_path=dataset_path)


print(dataset.class_indicies_distribution())

# image = dataset[0][0]

# cv2.imshow("Image", image)

# cv2.waitKey(0)