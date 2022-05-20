from re import I
from torch.nn import *
import torch
import torch.nn as nn
# 512 X 512



class BrainTumorCNN(torch.nn.Module):
    def __init__(self, number_of_classes=4, in_channel=3, image_size=512, drop_out_prob=0.5, batch_size=1):
        super(BrainTumorCNN, self).__init__()
        
        self.relu = nn.ReLU()
        
        # --- First layer ----------------------------------------------------------
        
        kernel_size, stride, padding = 3, 2, 1
        out_channels_1 = 32
        
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_1,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.max_pooling_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels_1)
        
        self.dropout_1 = nn.Dropout(p=drop_out_prob)
        
        # --- Second layer ---------------------------------------------------------
        
        kernel_size, stride, padding = 3, 1, 1
        out_channels_2 = 64
        
        self.conv_2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels_2)
        self.dropout_2 = nn.Dropout(p=0.25)
        
        
        # --- Weight initialization ------------------------------------------------
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)
        
    def forward(self, x):
        
        # --- First layer ----------------------------------------------------------
        x = self.conv_1(x)
        x = self.max_pooling_1(x)
        x = self.relu(x)
        
        x = self.batch_norm_1(x)
        x = self.dropout_1(x)
        
        # --- Second layer ---------------------------------------------------------
        
        x = self.conv_2(x)
        x = self.relu(x)
        
        x = self.batch_norm_2(x)
        x = self.dropout_2(x)
        
        # --- Third layer ----------------------------------------------------------
        
        return x
        
        
        
        