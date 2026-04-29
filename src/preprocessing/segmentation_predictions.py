import argparse
import ast
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import pydicom
import torch
import torchvision.transforms as T
import warnings
warnings.filterwarnings("ignore")

from src.models.dtunet import DTUNet
from .utils import remove_gray_image, cervix_caliper_concept, create_binary_mask, cervix_length_pred


def get_segmentations_and_cl_predictions(csv, checkpoint, output, root, gpu_id=0, plot=False):

    df = pd.read_csv(csv)

    os.environ["CUDA_VISIBLE_DEVICES"]= str(gpu_id)
    device = 'cpu' if torch.cuda.is_available() else 'cuda'
    
    tfs = T.Compose(
        [
            T.ToTensor(),
            T.Resize((224, 288)),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    model = DTUNet(1, 14).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    df['mask_dir'] = pd.Series()
    df['left'] = pd.Series()
    df['right'] = pd.Series()
    df['binary_mask_exists'] = pd.Series()
    df['canal_mask_exists'] = pd.Series()

    folder = 'SegMasks'
    root_mask_path = root + f'{folder}/{output}'
    os.makedirs(root_mask_path, exist_ok=True)

    for i in range(len(df)):
    
        device_name = df['device_name'][i]
        acceptable = df['acceptable'][i]
        px_spacing = df['px_spacing'][i]
        py_spacing = df['py_spacing'][i]
        img_calipers_path = df['image_dir_calipers'][i]
        img_path = img_calipers_path
        image_name = img_path.split('/')[-1]
        image_id =  image_name.split('.jpg')[0]

        image = Image.open(img_path)
        image = tfs(image).to(device)
        image = image.unsqueeze(0)
        image = torch.mean(image, dim=1, keepdims=True)

        print(i, image_id, px_spacing)

        with torch.no_grad():
            outputs = model({'image':image})
            logits = outputs['logit']

        pred = torch.argmax(logits, dim=1).squeeze()
        left, right = cervix_caliper_concept(pred.unsqueeze(0))
        

        logits = logits[:, :5, : ,: ].squeeze(0).detach().cpu().numpy()
        pred = pred.detach().cpu().numpy()
        pred[pred>4] = 0
        # pred = morphology.remove_small_objects(pred, min_size=128)

        if math.isnan(px_spacing):
            temp_df = df[df['device_name'] == df['device_name'][i]].px_spacing
            px_spacing = np.mean(temp_df)
            if math.isnan(px_spacing):
                px_spacing = np.mean(df.px_spacing.dropna())

        if math.isnan(py_spacing):
            temp_df = df[df['device_name'] == df['device_name'][i]].py_spacing
            py_spacing = np.mean(temp_df)
            if math.isnan(py_spacing):
                py_spacing = np.mean(df.py_spacing.dropna())

        binary_pred = create_binary_mask(pred, pixel_spacing=py_spacing*10, radius_in_mm=4)
        labels = np.unique(pred).tolist()
        binary_labels = np.unique(binary_pred).tolist()
        pred_mask_path = root_mask_path + f'/{image_id}'

        masks = {}
        masks['binary'] =  binary_pred
        masks['zahra'] = pred
        masks['logits'] = logits
        df['left'][i] = str(left)
        df['right'][i] = str(right)
        df['binary_mask_exists'][i] = 1 if 1 in binary_labels else 0
        df['canal_mask_exists'][i] = 1 if 1 in labels else 0
        df['mask_dir'][i] = pred_mask_path + '.npy'
        np.save(pred_mask_path, masks)
        
        if plot: # debugging plot
            # print(i, image_id, px_spacing)
            image = image.squeeze().detach().cpu().numpy()
            image = image / 2 + 0.5
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(image, cmap='gray')
            plt.subplot(1,3,2)
            plt.scatter(left[0], left[1], s=20, color='green')
            plt.scatter(right[0], right[1], s=20, color='red')
            plt.imshow(pred)
            plt.subplot(1,3,3)
            plt.imshow(binary_pred)
            plt.show()
    
    # compute cervical length based on segmentation predictions
    df['left'] = df['left'].map(ast.literal_eval)
    df['right'] = df['right'].map(ast.literal_eval)
    df['cervical_length'] = df.apply(cervix_length_pred, axis=1)

    updated_csv_name = csv.split('.csv')[0] + '_seg_pred' + '.csv'
    df.to_csv(f'{updated_csv_name}', index=False)     

    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Save segmentation predictions and cervical length estimates")
    parser.add_argument("--csv", help="path to csv", type=str, default='/home/ppar/SA-SonoNet-sPTB/metadata/dataset_preterm.csv')
    parser.add_argument("--checkpoint", help="path to segmentation model weights", type=str, default='/home/ppar/SA-SonoNet-sPTB/src/models/weights/dtunet/model.t7')
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--root", help="root path to save segmentation predictions", type=str)
    parser.add_argument("--output", help="output file and paths", type=str, default='preterm_birth_masks', choices=['preterm_birth_masks, term_birth_masks'])
    args = parser.parse_args() 
    
    
    df = get_segmentations_and_cl_predictions(csv=args.csv, checkpoint=args.checkpoint, output=args.output, root=args.root, gpu_id=args.gpu_id, plot=False)