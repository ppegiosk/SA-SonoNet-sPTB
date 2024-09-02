import argparse
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import pydicom
import scipy.ndimage
import skimage
import warnings
warnings.filterwarnings("ignore")


def remove_confounding_factors(img, grayscale_colormap=True):
    """
    Original implementation provided by Kamil Mikolaj
    If you find this function useful for your research please consider citing Mikolaj et al. [1]
    [1] "Removing confounding information from fetal ultrasound images.", arXiv preprint, arXiv:2303.13918 (2023)

    Preprocess ultrasound image by removing confounders like text and calipers

    Parameters:
    - img : Input RGB image (H, W, 3)
    - grayscale_colormap : Handle also ultrasound RGB images with grayscale colormap (default True)

    Returns:
    - Processed RGB image (H, W, 3)
    """
    # ensure input is a valid RGB image
    assert img.shape[-1] == 3, "image must have 3 channels (RGB)"
    img = copy.deepcopy(img)

    # apply thresholding in hue space to identify text
    if grayscale_colormap:
        roi_mask = img.sum(axis=2) > (10 if img.max() > 1 else 10/255.0)
    else:
        roi_mask = (skimage.color.rgb2hsv(img)[:, :, 0] < 0.15)

    roi_mask = skimage.morphology.dilation(roi_mask)
    roi_mask = get_largest_connected_component(scipy.ndimage.binary_fill_holes(roi_mask))

    # mask out everything outside the field of view
    img[~roi_mask] = 0 

    # remove calipers using telea method
    img = remove_calipers_telea(img)

    return img

def get_largest_connected_component(mask):
    """
    Get the largest connected component from a binary mask.

    Parameters:
    - mask : Binary mask

    Returns:
    - Largest connected component mask
    """
    labels = skimage.measure.label(mask)
    if labels.max() == 0:  # if no components, return full mask
        return np.ones_like(mask, dtype=bool)
    
    largest_label = np.argmax(np.bincount(labels.flat)[1:]) + 1
    return labels == largest_label

def remove_calipers_telea(img, radius=7):
    """
    Remove calipers using Telea inpainting algorithm.

    Parameters:
    - img : RGB image
    - radius : Inpainting radius (default 7)

    Returns:
    - Image with calipers removed
    """
    hsv = skimage.color.rgb2hsv(img)
    caliper_mask = ((0.11 < hsv[:, :, 0]) & (hsv[:, :, 0] < 0.17) & 
                    (hsv[:, :, 1] > 0.5) & (hsv[:, :, 2] > 0.5))
    caliper_mask = skimage.morphology.dilation(caliper_mask, skimage.morphology.square(radius))
    return cv2.inpaint(img, caliper_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)

def create_clean_dataset(csv, root, plot=False):

    df = pd.read_csv(csv)
    df = df.reset_index(drop=True)
    
    df['dicom_video'] = pd.Series()
    df['dicom_grayscale'] = pd.Series()
    df['image_dir_clean'] = pd.Series()
    df['image_dir'] = pd.Series()
    
    for i in range(len(df)):

        dicom_dir =  df['dicom_dir'][i]
        device_name = df['device_name'][i]
        image_name = dicom_dir
        image_name = image_name[1:] if image_name[0]=="/" else image_name
        image_name = image_name.replace("/","_")

        if "dcm" in image_name:
            image_name = image_name.replace(".dcm",".jpg")
        else:
            image_name = image_name + ".jpg"

        print(i, image_name)

        image_name_clean = image_name.replace(".jpg","_clean.jpg")

        # read dicom image
        img = pydicom.dcmread(dicom_dir).pixel_array

        if len(img.shape)==4:
            img = img[0,:,:,:]
            df['dicom_video'][i] = 1
        else:
            df['dicom_video'][i] = 0
                
        if len(img.shape)==3:
            df['dicom_grayscale'][i] = 0
            # remove patient information  
            img[:70] = 0
            # remove confounders
            img_clean = remove_confounding_factors(img, grayscale_colormap=True)
        else:
            # manual text removal for old ultrasound machines with grayscale signle channel images
            df['dicom_grayscale'][i] = 1
            img_clean = np.stack((img,) * 3, axis=-1)
            # remove patient information
            img[:70] = 0
            # device name can be retrieved from dicom files
            if device_name == 'iU22':
                if img_clean.shape[0]==768:
                    img_clean[:80] = 0
                    img_clean[-50:] = 0
                    img_clean[70:240, 0:120] = 0
                    img_clean[70:255, -120:] = 0
                    img_clean[130:175, 285:330] = 0
                    img_clean[-100:, -200:] = 0
                else:
                    img_clean[:80] = 0
            elif device_name == 'LOGIQ7':
                if img_clean.shape[0] == 480:
                    img_clean[:50] = 0
                    img_clean[-30:] = 0
                    img_clean[-50:, :100] = 0
                    img_clean[:, -110:] = 0
                elif img_clean.shape[0] == 768:
                    img_clean[:80] = 0
                    img_clean[-50:] = 0
                    img_clean[:, -200:] = 0
            elif device_name == 'LOGIQ5':
                img_clean[:50] = 0
                img_clean[-20:] = 0
                img_clean[50:255, -120:] = 0
                img_clean[45:65, 175:205] = 0
                img_clean[-50:, :100] = 0

        os.makedirs(root + '/clean/', exist_ok=True)
        os.makedirs(root + '/calipers/', exist_ok=True)
        image_dir_clean = root + '/clean/' + image_name_clean
        image_dir_calipers = root + '/calipers/' + image_name
        df['image_dir_clean'][i] = image_dir_clean
        df['image_dir'][i] = image_dir_calipers

        # save image with and without counfounders as jpg images
        img = Image.fromarray(img)
        img.save(image_dir_calipers)
        img_clean = Image.fromarray(img_clean)
        img_clean.save(image_dir_clean)

        if plot: # debugging plot
            plt.subplot(1,2,1)
            plt.imshow(img)
            plt.subplot(1,2,2)
            plt.imshow(img_clean)
            plt.show()
        
    updated_csv_name = csv.split('.csv')[0] + '_confounders_removed' + '.csv'
    df.to_csv(f'{updated_csv_name}', index=False)

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser('Remove confounders from fetal ultrasound dicom images')
    parser.add_argument('--csv', type=str, help='path to dataset csv with input dicom image paths', default='/home/ppar/SA-SonoNet-sPTB/metadata/dataset_preterm.csv')
    parser.add_argument('--root', type=str, help='root path to save jpg images')
    args = parser.parse_args()
    
    df = create_clean_dataset(args.csv, root=args.root, plot=False)