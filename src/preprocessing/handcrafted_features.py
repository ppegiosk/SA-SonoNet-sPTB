import argparse
import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
import pyfeats
import warnings
warnings.filterwarnings('ignore')

def compute_handcrafted_features(labels, hist_equal=False):
    IMG_NO = len(labels)

    # parameters
    perc = 1                    # percentage of the plaque to take into consideration when calculating features in (0,1]
    Dx             = [0,1,1,1]  # early textural - GLDS
    Dy             = [1,1,0,-1] # early textural  - GLDS
    d              = 1          # early textural  - NGTDM
    Lr, Lc         = 4, 4       # early textural  - SFM
    l              = 7          # early textural  - LTE
    s              = 4          # early textural  - FDTA
    th             = [135,140]  # late textural - HOS
    P              = [8,16,24]  # late textural - LBP
    R              = [1,2,3]    # late textural - LBP

    # Init arrays
    names = []
    np_fos = np.zeros((IMG_NO,16), np.double)
    np_glcm_mean = np.zeros((IMG_NO,14), np.double)
    np_glcm_range = np.zeros((IMG_NO,14), np.double)
    np_glds = np.zeros((IMG_NO,5), np.double)
    np_ngtdm = np.zeros((IMG_NO,5), np.double)
    np_sfm = np.zeros((IMG_NO,4), np.double)
    np_lte = np.zeros((IMG_NO,6), np.double)
    np_fdta = np.zeros((IMG_NO,s+1), np.double)
    np_glrlm = np.zeros((IMG_NO,11), np.double)
    np_fps = np.zeros((IMG_NO,2), np.double)
    np_hos = np.zeros((IMG_NO,len(th)), np.double)
    np_lbp = np.zeros((IMG_NO,len(P)*2), np.double)
    np_glszm = np.zeros((IMG_NO,14), np.double)

    # Calculate Features
    progress = tqdm(range(0,IMG_NO), desc="Calculating Textural Features...")
    for i in progress:

        name = labels['dicom_dir'][i]
        names.append(name)

        image_dir_clean = labels['image_dir_clean'][i]
        binary_mask_dir = labels['mask_dir'][i]

        mask_data = np.load(binary_mask_dir, allow_pickle=True).item()
        mask = mask_data['binary']
        
        img = Image.open(image_dir_clean)
        img = img.resize((288,224))
        img = ImageOps.grayscale(img)
        img = np.asarray(img)

        mask = np.asarray(mask)
        mask[mask>0.5]=1
        mask[mask<=0.5]=0
        
        if hist_equal:
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            img = clahe.apply(img)

        # textural Features
        progress.set_description('Calculating Early Textural Features' + ' for ' + name)
        np_fos[i,:], labels_fos = pyfeats.fos(img, mask)
        np_glcm_mean[i,:], np_glcm_range[i,:], labels_glcm_mean, labels_glcm_range = pyfeats.glcm_features(img, ignore_zeros=False)
        np_glds[i,:], labels_glds = pyfeats.glds_features(img, mask, Dx=Dx, Dy=Dy)
        np_ngtdm[i,:], labels_ngtdm = pyfeats.ngtdm_features(img, mask, d=d)
        np_sfm[i,:], labels_sfm = pyfeats.sfm_features(img, mask, Lr=Lr, Lc=Lc)
        np_lte[i,:], labels_lte = pyfeats.lte_measures(img, mask, l=l)
        np_fdta[i,:], labels_fdta = pyfeats.fdta(img, mask, s=s) 
        np_glrlm[i,:], labels_glrlm = pyfeats.glrlm_features(img, mask, Ng=256)
        np_fps[i,:], labels_fps = pyfeats.fps(img, mask)
        progress.set_description('Calculating Late Textural Features' + ' for ' + name)
        np_hos[i,:], labels_hos = pyfeats.hos_features(img, th=th)
        np_lbp[i,:], labels_lbp = pyfeats.lbp_features(img, mask, P=P, R=R)
        np_glszm[i,:], labels_glszm = pyfeats.glszm_features(img, mask)
            
    # early textural features
    df_fos = pd.DataFrame(data=np_fos, index=names, columns=labels_fos)
    df_glcm_mean = pd.DataFrame(data=np_glcm_mean, index=names, columns=labels_glcm_mean)
    df_glcm_range = pd.DataFrame(data=np_glcm_range, index=names, columns=labels_glcm_range)
    df_glds = pd.DataFrame(data=np_glds, index=names, columns=labels_glds)
    df_ngtdm = pd.DataFrame(data=np_ngtdm, index=names, columns=labels_ngtdm)
    df_sfm = pd.DataFrame(data=np_sfm, index=names, columns=labels_sfm)
    df_lte = pd.DataFrame(data=np_lte, index=names, columns=labels_lte)
    df_fdta = pd.DataFrame(data=np_fdta, index=names, columns=labels_fdta)
    df_glrlm = pd.DataFrame(data=np_glrlm, index=names, columns=labels_glrlm)
    df_fps = pd.DataFrame(data=np_fps, index=names, columns=labels_fps)

    # late textural features
    df_hos = pd.DataFrame(data=np_hos, index=names, columns=labels_hos)
    df_lbp = pd.DataFrame(data=np_lbp, index=names, columns=labels_lbp)
    df_glszm = pd.DataFrame(data=np_glszm, index=names, columns=labels_glszm)

    df_texture_all = pd.concat([df_fos, df_glcm_mean, df_glcm_range, df_glds,
                                df_ngtdm, df_sfm, df_lte, df_fdta, df_glrlm, df_fps,
                                df_hos, df_lbp, df_glszm], axis=1)

    df_texture_all = df_texture_all.dropna(axis=1)
    df_texture_all = df_texture_all.loc[:, df_texture_all.apply(pd.Series.nunique) !=1]
    feature_names = df_texture_all.columns.tolist()
    labels = labels.set_index('dicom_dir', drop=False)
    df = pd.merge(labels, df_texture_all,  left_index=True, right_index=True)
    df = df.reset_index(drop=True)
    return df, feature_names

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Extract handcratfed textural feautures')
    parser.add_argument('--path_to_features', type=str, default='/home/ppar/SA-SonoNet-sPTB/metadata', help='path to load pre-calculated features')
    parser.add_argument('--csv', type=str, default='/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_dataset_splits.csv', help='path to csv with input image paths')
    parser.add_argument('--hist_equal', action='store_true', help='apply adaptive historogram equalization before extracting features')
    args = parser.parse_args()

    os.makedirs(args.path_to_features, exist_ok=True)
    labels = pd.read_csv(args.csv)
    df, feature_names = compute_handcrafted_features(labels=labels, hist_equal=args.hist_equal)

    # save with pandas
    try:
        df.to_csv(args.path_to_features + '/ASMUS_MICCAI_texture_dataset_splits.csv', index = False)
        with open(f"{args.path_to_features}/textural_features.txt", "w") as output:
            output.write(str(feature_names))
        print('\nData was successfully saved')
    except:
        print("\nAn exception occured")