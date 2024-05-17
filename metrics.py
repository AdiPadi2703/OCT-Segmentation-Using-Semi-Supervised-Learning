# This file contains all the functions for evaluating iou and dice coefficient

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

####################################################################################################################
'''
multiclass intersection over union

-> based on the implementation used in the torchmetrics library for MeanIoU
-> both inputs are of the shape (batch_size, height, width) with the classes as integers
'''

def iou(preds, target, num_classes, SMOOTH=1e-7):
        
    preds = torch.nn.functional.one_hot(preds, num_classes).permute(3, 0, 1, 2)
    target = torch.nn.functional.one_hot(target, num_classes).permute(3, 0, 1, 2)
    
    reduce_axis = list(range(2, preds.ndim))
    intersection = torch.sum(preds & target, reduce_axis)
    target_sum = torch.sum(target, reduce_axis)
    pred_sum = torch.sum(preds, reduce_axis)
    union = target_sum + pred_sum - intersection
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    
    return torch.mean(iou,0)

######################################################################################################################
'''
multiclass dice coefficient for 4 classes (can be modified for 'n' classes by changing 4 to 'num_classes' 
                                           and adding function argument 'num_classes')

-> outputs dice coefficient for each class
-> pred shape is (batch_size, num_classes, height, width)
-> target shape is (batch_size, height, width) with classes as integers
'''

def multiclass_dice(pred,target):
    smooth = 1e-8
    target=F.one_hot(target,num_classes=4)
    a_sum=torch.sum(pred,dim=3)
    a_sum=torch.sum(a_sum,dim=2)
    a_sum=torch.sum(a_sum,dim=0).view(-1)
    
    b_sum=torch.sum(target,dim=1)
    b_sum=torch.sum(b_sum,dim=1)
    b_sum=torch.sum(b_sum,dim=0).view(-1)
    reshaped_target=torch.permute(target,(0,3,1,2))
    
    intersect= pred * reshaped_target

    intersect_sum=torch.sum(intersect,dim=3)
    intersect_sum=torch.sum(intersect_sum,dim=2)
    intersect_sum= torch.sum(intersect_sum,dim=0).view(-1)
    class_wise_dice=torch.div(intersect_sum*2+smooth,a_sum+b_sum+smooth)

    del(a_sum)
    del(b_sum)
    del(reshaped_target)
    del(intersect)
    del(intersect_sum)

    return (class_wise_dice)

#########################################################################################################################