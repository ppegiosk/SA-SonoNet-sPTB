import albumentations as A
import argparse
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import torch

from src.datamodules.pretermbirth_image_dataset import PretermBirthImageDataset
from src.models.sasononet import SASonoNetModel, SASonoNet

import src.util as util
from src.preprocessing.utils import cervix_caliper_concept, create_binary_mask
from src.pretermbirth_model import PretermBirthModel


def load_model(logdir, hparams):
    model = PretermBirthModel(hparams)
    checkpoint_path = util.get_checkoint_path_from_logdir(logdir)
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    
    if 'sa-sononet' in hparams.model.lower():
        if any('image_features' in key for key in state_dict.keys()):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('image_features', 'feature_extractor') if 'image_features' in key else key
                new_state_dict[new_key] = state_dict[key]
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    elif 'mt-unet' in hparams.model.lower():
        if any('fc' in key for key in state_dict.keys()):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('fc', 'classifier') if 'fc' in key else key
                new_state_dict[new_key] = state_dict[key]
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
    elif 'texturenet' in hparams.model.lower():
        if any('fc' in key for key in state_dict.keys()):
            new_state_dict = {}
            for key in state_dict.keys():
                new_key = key.replace('fc', 'texture_classifier') if 'fc' in key else key
                new_state_dict[new_key] = state_dict[key]
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    model.eval()
    return model, hparams


def load_torch_state_dict(checkpoint_path):

    checkpoint = torch.load(checkpoint_path)
    updated_state_dict = {}
    state_dict = checkpoint['state_dict']
    for key, value in state_dict.items():
        new_key = key.split('net.')[-1] 
        updated_state_dict[new_key] = value

    return updated_state_dict

def load_sasononet(checkpoint_path):

    model = SASonoNetModel(config='SN32', in_channels=8, num_labels=1)
    state_dict = load_torch_state_dict(checkpoint_path)
    dtunet_state_dict = torch.load(model.dtunet_checkpoint)

    new_state_dict = {}
    for key in dtunet_state_dict.keys():
        new_state_dict[f'dtunet.{key}'] = dtunet_state_dict[key]
    for key in state_dict.keys():
        new_key = key.replace('image_features', 'feature_extractor') if 'image_features' in key else key
        new_state_dict[new_key] = state_dict[key]
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run SA-SonoNet model")
    parser.add_argument("--csv", help="path to csv", type=str, default='/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_dataset_splits.csv')
    parser.add_argument("--label", help="label name", type=str, default='birth_before_week_37')
    parser.add_argument("--path", help="path to model checkpoint to initialize with", type=str, default='/home/ppar/SA-SonoNet-sPTB/models/sa-sononet/version_0/')
    parser.add_argument("--split_index", help="data split index for testing", type=str, default='fold_1')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # hparams = util.load_hparams_from_logdir(path)
    # sasononet, hparams = load_model(path, hparams)
    # sasononet.eval()

    checkpoint_path = util.get_checkoint_path_from_logdir(args.path)
    sasononet = load_sasononet(checkpoint_path)
    
    tfs = [A.Resize(224, 288)]
    testdata = PretermBirthImageDataset(csv_dir=args.csv, split_index=args.split_index, label_name=args.label,  transforms=tfs, split='test')
    loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False)
    print(f"test data: {len(testdata)}")

    for i, data in enumerate(loader):
        if i==10:break
        image, image_with_calipers, label, binary_mask = data['image'], data['image_with_calipers'], data['label'], data['binary_mask']
        spacing = data['spacing']
        segmentation_mask = data['segmentation_mask']
        segmentation_logits = data['segmentation_logits']

        output = sasononet(data)
        logit = output['logit'].squeeze(-1)
        prediction = torch.sigmoid(logit)
        mask_logit = output['mask']
        dtunet_mask = torch.argmax(mask_logit, dim=1).squeeze()

        left, right = cervix_caliper_concept(dtunet_mask.unsqueeze(0))
        inner_x, inner_y = left
        external_x, external_y = right
        px_spacing = torch.unique(spacing[:, 0, :, :]).item()
        py_spacing = torch.unique(spacing[:, 1, :, :]).item()
        cervical_length = math.sqrt(((inner_x-external_x) * px_spacing )**2 +( (inner_y-external_y) * py_spacing)**2)
        binary_mask = create_binary_mask(dtunet_mask, pixel_spacing=py_spacing*10, radius_in_mm=4)


        cl_text = f'CL = {np.round(cervical_length, 2)} mm'
        risk_text = f'Risk Score = {np.round(np.round(prediction.item(), 3) * 100, 1)} %'
        label_text =f'True Label = {int(label.item())}'

        print()
        print(cl_text)
        print(risk_text)
        print(label_text)
        print()

        plt.figure(figsize=(12,4))
        plt.suptitle(label_text, fontsize=16, y=0.95)

        plt.subplot(1,3,1)
        plt.title(cl_text)
        plt.imshow((image.squeeze(0).squeeze(0).numpy()+1)/2, cmap='gray')
        plt.scatter(left[0], left[1], s=10, color='red')
        plt.scatter(right[0], right[1], s=10, color='red')
        plt.plot([left[0], right[0]], [left[1], right[1]], color='red')
        plt.axis('off')

        plt.subplot(1,3,2)
        plt.title(risk_text)
        plt.imshow(dtunet_mask.squeeze(0))
        plt.axis('off')

        plt.subplot(1,3,3)
        plt.title('Binary Mask Output')
        plt.imshow((binary_mask+1)/2, cmap='gray')
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()