from bt_dataset import *
import cv2



dataset = BrainTumorDataset()

image = dataset[0][0]

cv2.imshow("Image", image)

cv2.waitKey(0)