from re import I
from torch.nn import *
import torch
import torch.nn as nn


class BrainTumorCNN(torch.nn.Module):
    def __init__(self, number_of_classes=4, in_channel=3, image_size=256, drop_out_prob=0.5):
        super(BrainTumorCNN, self).__init__()
        
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        # --- First layer ----------------------------------------------------------
        
        kernel_size, stride, padding = 3, 2, 1
        out_channels_1 = 32
        
        self.conv_1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channels_1,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        # --- Feature map size after 1st convolution:

        final_image_size = feature_map_size(image_size, kernel_size=kernel_size,
                                            stride=stride, padding=padding)

        # --- Feature map size after 1st pooling:

        self.max_pooling_1 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

        final_image_size = feature_map_size(final_image_size, kernel_size=kernel_size,
                                            stride=kernel_size, padding=0)
        
        self.batch_norm_1 = nn.BatchNorm2d(num_features=out_channels_1)
        self.dropout_1 = nn.Dropout(p=drop_out_prob)
        
        # --- Second layer ---------------------------------------------------------
        
        kernel_size, stride, padding = 3, 2, 1
        out_channels_2 = 64
        
        self.conv_2 = nn.Conv2d(in_channels=out_channels_1, out_channels=out_channels_2,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.batch_norm_2 = nn.BatchNorm2d(num_features=out_channels_2)
        self.dropout_2 = nn.Dropout(p=drop_out_prob)
        
        # --- Feature map size after 2nd convolution:
        final_image_size = feature_map_size(final_image_size, kernel_size=kernel_size,
                                            stride=stride, padding=padding)

        # --- Feature map size after 2nd pooling:
        self.max_pooling_2 = nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)

        final_image_size = feature_map_size(final_image_size, kernel_size=kernel_size,
                                            stride=kernel_size, padding=0)

        # --- Third layer ---------------------------------------------------------
        
        kernel_size, stride, padding = 3, 1, 1
        out_channels_3 = 128
        
        self.conv_3 = nn.Conv2d(in_channels=out_channels_2, out_channels=out_channels_3,
                                kernel_size=kernel_size, stride=stride, padding=padding)
        
        self.batch_norm_3 = nn.BatchNorm2d(num_features=out_channels_3)
        self.dropout_3 = nn.Dropout(p=drop_out_prob)
        
        # --- Feature map size after 3rd convolution:

        final_image_size = feature_map_size(final_image_size, kernel_size=kernel_size,
                                            stride=stride, padding=padding)
        
        # --- Fully Connected layer ------------------------------------------------

        in_features = final_image_size ** 2 * out_channels_3 
        self.fc_1 = nn.Linear(in_features=in_features, out_features=number_of_classes)

        # --- Weight initialization ------------------------------------------------
        
        nn.init.kaiming_normal_(self.conv_1.weight)
        nn.init.kaiming_normal_(self.conv_2.weight)
        nn.init.kaiming_normal_(self.conv_3.weight)
        
        # --- Bias initialization -------------------------------------------------
        
        nn.init.constant_(self.conv_1.bias, 0.0)
        nn.init.constant_(self.conv_2.bias, 0.0)
        nn.init.constant_(self.conv_3.bias, 0.0)
        
        # -------------------------------------------------------------------------
        
    def forward(self, x):
        
        # --- First layer ----------------------------------------------------------

        x = self.conv_1(x)
        x = self.batch_norm_1(x)

        x = self.max_pooling_1(x)
        x = self.leaky_relu(x)
        x = self.dropout_1(x)
        
        # --- Second layer ---------------------------------------------------------
        
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.max_pooling_2(x)
        x = self.leaky_relu(x)
        x = self.dropout_2(x)
        
        # --- Third layer ----------------------------------------------------------
        
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.leaky_relu(x)
        x = self.dropout_3(x)
        
        # --- Fully Connected layer -------------------------------------------------

        x = x.view(x.size(0), -1)

        x = self.fc_1(x)
        
        return x
        
        
def feature_map_size(image_size : int, kernel_size : int,
                     stride : int, padding : int):
    new_feature_map_size = (image_size - kernel_size + 2 * padding) // stride + 1
  
    return new_feature_map_size
    