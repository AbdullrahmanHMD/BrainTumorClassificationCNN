from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from data.preprocessing import *

DEFULT_PATH = os.getcwd()
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class BrainTumorDataset(Dataset):
    
    def __init__(self, dataset_path=DEFULT_PATH, test=False, preprocessing=None):
        
        if test:
            dataset_type = 'Testing'
        else:
            dataset_type = 'Training'
        
        self.preprocessing = preprocessing
        
        self.dataset_path = os.path.join(dataset_path, dataset_type)
        
        # --- Dataset general specification -----------------------
        self.image_size = 512
        
        # The labels of the classes in the dataset:
        self.classes = os.listdir(self.dataset_path)
        self.num_classes = len(self.classes)
        
        # The labels of the classes in the dataset mapped to integers:
        self.mapped_classes = self.get_classes()
        
        # ----------------------------------------------------------
        # Loading data paths and its labels:
        self.data_paths, self.data_labels, self.data_labels_txt = self.load_dataset_paths()
        
        # The distribution of the data points among classes in the dataset:
        self.data_distribution = self.get_data_distribution()
        
        # ----------------------------------------------------------
        # Class weights for the weighted loss:
        self.class_weights = self.calculate_class_weights()

    def get_classes(self):
        classes = {}
        for i, class_ in enumerate(os.listdir(self.dataset_path)):
            classes[class_] = i
        
        return classes
    
    
    def load_dataset_paths(self):
        data_labels = []; data_labels_txt = []; data_paths = []
        
        for label, label_txt in enumerate(os.listdir(self.dataset_path)):
            class_path = os.path.join(self.dataset_path, label_txt)
            for datum in os.listdir(class_path):
                datum_path = os.path.join(class_path, datum)
                
                data_labels.append(label)
                data_labels_txt.append(label_txt)
                data_paths.append(datum_path)
        
        return data_paths, np.array(data_labels), np.array(data_labels_txt)
                
               
    def get_data_distribution(self):
        data_distribution = np.bincount(self.data_labels)
        
        data_distribution_dict = {}
        
        for label_txt, dist in zip(self.classes, data_distribution):
            data_distribution_dict[label_txt] = dist
        return data_distribution_dict
        
        
    def plot_data_distribution(self):
        x = self.classes
        y = np.bincount(self.data_labels)
        
        fig, ax = plt.subplots()
        bars = ax.barh(x, y)
        ax.bar_label(bars)
        
        plt.show()
    
    def calculate_class_weights(self):
        
        bin_count = np.bincount(self.data_labels)
        dataset_size = len(self.data_labels)
        
        class_weights = [1 - (class_ / dataset_size) for class_ in bin_count]
        
        return class_weights


    def class_indicies_distribution(self):
        indicies_distribution = {}
        for class_ in self.mapped_classes.values():
            mask = class_ == self.data_labels
            indicies_distribution[class_] = [i for i, x in enumerate(mask) if x]
        return indicies_distribution


    def __getitem__(self, index):
        image_path = self.data_paths[index]
        image = cv2.imread(image_path)
        
        image_label = self.data_labels[index]
        image_label_txt = self.data_labels_txt[index]
        if self.preprocessing != None:
            image = self.preprocessing(image)
        else:
             image = torch.tensor(image)

        return image, image_label, image_label_txt
        
    
    def __len__(self):
        return len(self.data_labels)