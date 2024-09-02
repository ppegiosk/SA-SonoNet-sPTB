import cv2
import math
import numpy as np
import pandas as pd
from PIL import Image
import skimage
from skimage.filters import threshold_otsu
import torch
from torchvision.io import read_image

def cervix_caliper_concept(seg_mask):
    """
    Get left and right-most points from the cervical canal (CC) segmentation
    These are used to estimate cervical length 

    Parameters:
    - mask : DTU-Net's segmentation mask

    Returns:
    -  Left and right-most points from CC (channel id 1) 
    """

    for b in range(seg_mask.size(0)):
        mask = seg_mask[b, ...].unsqueeze(-1)
        x_right, x_left, y_right, y_left = 0,0,0,0
        if torch.sum(mask==1) >= 50:
            points = torch.nonzero(mask==1)
            index_left, index_right = torch.argmin(points[:,1]), torch.argmax(points[:,1])
            x_left, x_right, y_left, y_right = points[index_left, 0], points[index_right, 0], points[index_left, 1], points[index_right, 1]
            
            length = y_right - y_left

            index = points[:,1]< y_left+length/100
            left_points = points[index, :]
            height = (torch.max(left_points[:,0].float()) + torch.min(left_points[:,0].float()))/2 
            index = points[:,1]< y_left+length/100
            left_points = points[index, :]
            index = torch.argmin(torch.abs(height - left_points[:, 0]))
            x_left, y_left, _ = left_points[index, :]

            index = points[:,1]> y_right-length/100
            right_points = points[index, :]
            height = (torch.max(right_points[:,0].float()) + torch.min(right_points[:,0].float()))/2
            index = points[:,1]> y_right-length/100
            right_points = points[index, :]
            index = torch.argmin(torch.abs(height - right_points[:, 0]))
            x_right, y_right, _ = right_points[index, :]

            x_left, x_right, y_left, y_right = x_left.item(), x_right.item(), y_left.item(), y_right.item()

    left = [y_left, x_left]
    right = [y_right, x_right]

    return left, right

def create_binary_mask(mask, radius_in_mm=3, pixel_spacing=0.5):
    """
    Create a binary segmentation mask around cervical canal (CC) segmentation similar to Włodarczy et. al. [1, 2]
    [1] Spontaneous preterm birth prediction using convolutional neural networks 
    [2] Estimation of preterm birth markers with U-net segmentation network
    This is used to extract handcrafted features around CC

    Parameters:
    - mask : DTU-Net's segmentation mask
    - radium_in_mm : radius to apply morphological dilation
    - pixel_spacing : pixel spacing to compute radius

    Returns:
    -  Binary mask 
    """

    mask = np.where(mask!=1, 0, 255)
    mask = mask.astype(np.uint8)

    radius = math.floor(radius_in_mm / pixel_spacing)
    strd = skimage.morphology.disk(radius)
    binary_mask = skimage.morphology.dilation(mask, strd)
    binary_mask = cv2.GaussianBlur(binary_mask, (0,0), sigmaX=7, sigmaY=7, borderType=cv2.BORDER_DEFAULT)

    thresh = threshold_otsu(binary_mask)
    binary_mask = binary_mask > thresh

    binary_mask = binary_mask.astype(np.int64)

    return binary_mask

def cervix_length_pred(row):
    spacing_x = row.px_spacing * 10
    spacing_y = row.py_spacing * 10
    inner_x, inner_y = row['left']
    external_x, external_y = row['right']
    cl = math.sqrt(((inner_x-external_x) * spacing_x )**2 +( (inner_y-external_y) * spacing_y)**2)
    return cl

def remove_gray_image(x):
    img = read_image(x)
    if img.shape[0] == 1:
        return False
    elif img.shape[0] == 3:
        return True