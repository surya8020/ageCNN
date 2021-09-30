import segmentation_models_pytorch as smp
import albumentations as albu
import torch
import utils as utils
import torchinfo as torchinfo
import os
from torchvision.io import read_image
import numpy as np


def testFunc(x):
    '''
    Implement the sigmoid function using the formula given in the proj2 notebook. 
    You are not allowed to use any pre-defined sigmoid function from any library. 

    Args:
        x: N x 1 torch.FloatTensor

    Returns:
        y: N x 1 torch.FloatTensor
    '''
    y = x**2

    return y
