import numpy as np
import torchvision
import cv2
import torch


def resize(size):
    return lambda image : cv2.resize(image, size)

def sharpen():
    sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1 , 0]])
    return lambda image : cv2.filter2D(image, ddepth=-1, kernel=sharpening_kernel)
    
def denoise():
    return lambda image : cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def normalize():
    return lambda image : cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

def sequential_preprocessing(image : np.ndarray, preprocessing_ops : list):
    for preprocessing_op in preprocessing_ops:
        image = preprocessing_op(image)
        
    return image