# This file contains the model architecture. The student and teacher model both have the same architecture

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from torch import nn
import os
import random
from IPython.display import clear_output
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import Unet

#################################################################################################################################

class StudentModel(nn.Module):
    
    def __init__(self, in_channels, num_classes, encoder, encoder_weights):
        super(StudentModel, self).__init__()
        
        self.unet = Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes, activation='softmax')

    def forward(self, x):
        
        return self.unet(x)
    
#################################################################################################################################
    
    
class TeacherModel(nn.Module):
    
    def __init__(self, in_channels, num_classes, encoder, encoder_weights):
        super(TeacherModel, self).__init__()
        
        self.unet = Unet(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels, classes=num_classes, activation='softmax')
        
    def forward(self, x):
        
        return self.unet(x)
    
####################################################################################################################################