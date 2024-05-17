# This file contains all the functions for getting the consistency loss weight 

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import nn
import os
import random
import torch.nn.functional as F
import segmentation_models_pytorch as smp

''' Original Source : https://arxiv.org/abs/1610.02242'''

########################################################################

def sigmoid_rampup(current, rampup_length):

    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
#########################################################################

def get_current_consistency_weight(epoch):

    consistency = 10.0
    consistency_rampup = 5.0
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

##########################################################################