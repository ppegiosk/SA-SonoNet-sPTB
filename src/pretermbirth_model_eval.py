import argparse
from glob import glob
import numpy as np
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import random
import torch
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, f1_score, precision_score, confusion_matrix
from torchmetrics.functional import auroc, recall, precision, specificity, accuracy

from src.pretermbirth_model import PretermBirthModel
import src.util as util
from src.evaluation.calibration import get_unbiased_calibration_rmse


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PretermBirthModel.add_model_specific_args(parser)
    parser = add_program_level_args(parser)
    return parser

def add_program_level_args(parent_parser):
    parser = parent_parser.add_argument_group("Program Level Arguments")
    parser.add_argument(
        "--path", help="path to model checkpoint to initialize with"
    )

    return parent_parser

def load_datamodule_from_params(hparams):
    datamodule = util.load_datamodule_from_name(
        dataset_name=hparams.dataset,
        batch_size=hparams.batch_size, 
        split_index=hparams.split_index,
        # img_size= hparams.img_size,
        label=hparams.label,
        )
    return datamodule, hparams

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

def main(args):

    model_paths = sorted(glob(f'{args.path}/*/'))

    auc_list, acc_list, recall_list, specificity_list, calibration_list = [],[],[], [], []

    for model_i, model_path in enumerate(model_paths):
        
        hparams = util.load_hparams_from_logdir(model_path)
        os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        print()

        experiment = str(model_path).split('/')[-3]
        experiment_path = f'/home/ppar/SA-SonoNet-sPTB/results/{experiment}'
        os.makedirs(experiment_path, exist_ok=True)

        model_name = experiment + '_'+ str(model_path).split('/')[-2]
        
        datamodule, hparams = load_datamodule_from_params(hparams)
        model, hparams = load_model(model_path, hparams)
        model.eval()
        model.to(device)

        dataloader = datamodule.test_dataloader()
        label_name = dataloader.dataset.label_name
        split_index = dataloader.dataset.split_index

        print('model name: ', model_name)
        print('split index', split_index)
        print('test set: ', datamodule.test_dataloader().dataset.__len__())

        test_predictions, test_logits, test_labels= [], [], []

        for i, batch in enumerate(dataloader):

            for key,value in batch.items():
                if torch.is_tensor(value):
                    batch[key]=value.to(device)

            with torch.no_grad():
                if 'mt-unet' in hparams.model.lower():
                    logits, _ = model(batch)
                else:
                    logits = model(batch)

                logits = logits.squeeze(-1)
                labels = batch['label'].float()
                preds = torch.sigmoid(logits)

                test_predictions.append(preds)
                test_logits.append(logits)
                test_labels.append(labels)
                
        pred = torch.concat(test_predictions).detach()
        label = torch.concat(test_labels)
        test_predictions = torch.concat(test_predictions).detach().cpu().numpy()
        test_labels = torch.concat(test_labels).unsqueeze(-1).detach().cpu().numpy()
        test_raw_predictions = test_predictions.copy()

        test_predictions[test_predictions>0.5] = 1
        test_predictions[test_predictions<=0.5] = 0
        test_predictions = test_predictions.astype(int)

        cm = confusion_matrix(test_labels, test_predictions)
        auc = roc_auc_score(test_labels, test_raw_predictions)
        acc = accuracy_score(test_labels, test_predictions)
        rec = recall_score(test_labels, test_predictions, pos_label=1)
        spec = recall_score(test_labels, test_predictions, pos_label=0)
        cal_t = get_unbiased_calibration_rmse(test_labels.reshape(-1), test_raw_predictions.reshape(-1))

        auc_list.append(auc)
        acc_list.append(acc)
        recall_list.append(rec)
        specificity_list.append(spec)
        calibration_list.append(cal_t)

        print(f'AUC: {np.round(auc, 3)}, ACC: {np.round(acc, 3)}, SEN: {np.round(rec, 3)}, SPE: {np.round(spec, 3)}, UCE: {np.round(cal_t, 3)}')
    
        df = pd.DataFrame({f'label': test_labels.reshape(-1), 
                            'prediction': test_predictions.reshape(-1),
                            'confidence': test_raw_predictions.reshape(-1), 
                        })
        
        assert (df['label'] == dataloader.dataset.csv['birth_before_week_37']).sum() == df.shape[0]
        df = pd.concat([df, dataloader.dataset.csv], axis=1)
        df['label'] = df['label'].astype(int)
        df.to_csv(f'{experiment_path}/{model_name}.csv', index=False)
    
    mean_acc, std_acc = np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3)
    mean_auc, std_auc = np.round(np.mean(auc_list), 3), np.round(np.std(auc_list), 3)
    mean_recall, std_recall = np.round(np.mean(recall_list), 3), np.round(np.std(recall_list), 3)
    mean_specificity, std_specificity = np.round(np.mean(specificity_list), 3), np.round(np.std(specificity_list), 3)
    mean_calibration, std_calibration = np.round(np.mean(calibration_list), 3), np.round(np.std(calibration_list), 3)

    print()
    print(f'Results - model: {experiment}')
    print(f'AUC: {mean_auc}±{std_auc}, ACC: {mean_acc}±{std_acc}, SEN: {mean_recall}±{std_recall}, SPE: {mean_specificity}±{std_specificity}, UCE: {mean_calibration}±{std_calibration}')
    
if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args)