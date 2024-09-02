import argparse
from datetime import datetime
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def cervical_length_results(data, label = 'birth_before_week_37'):
    cm = confusion_matrix(data[label], data['pred'])
    auc = 1 - roc_auc_score(data[label], data['cervical_length'])
    acc = accuracy_score(data[label], data['pred'])
    recall = recall_score(data[label], data['pred'], pos_label=1)
    specificity = recall_score(data[label], data['pred'], pos_label=0)
    return auc, acc, recall, specificity 

def drop_rows(df, term=True):
    df = df[df['image_dir_clean'].notna()]
    df = df[df['cervical_length'].notna()]
    df = df[df['bmi'].notna()]
    df = df[df['ga'].notna()]
    df = df[df['ga_at_birth'].notna()]
    
    df = df[df['cervical_length']!=0]
    df = df[df['binary_mask_exists']==1]
    
    df = df[df['days_to_birth']>=0]
    df = df[df['days_to_birth']<=150]
    
    df = df[df['trimester']!=1]
    df = df[df['ga_in_weeks']<=32]
    df = df[df['ga_in_weeks']>=19]

    df = df[df['device_name'] != 'Voluson S10']
    df = df[df['device_name'] != 'Antares']
    df = df[df['device_name'] != 'LOGIQ7']
    df = df[df['device_name'] != 'Voluson S6']
    
    if term:
        df= df[df['acceptable']==1]    
    return df

def stratified_split_patients(df, n_splits=5, stratify_by='ga_in_weeks', group='patient_id', seed=None, debug=False):
    if seed is None: 
        shuffle = False 
    else:
        shuffle = True
    y = df[stratify_by]
    X = df.drop(stratify_by, axis=1)
    # X = X.sample(frac=1, random_state=23).reset_index(drop=True)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    for i, (train_index, val_index) in enumerate(splitter.split(X=X,y=y,groups=df[group])):
        if debug:
            print(f"Fold {i}:")
            print(f"     Train: index={train_index}, len: {len(train_index)}")
            print(f"            group={df[group][train_index].values}")
            print(f"     Val:  index={val_index},  len: {len(val_index)}")
            print(f"            group={df[group][val_index].values}")
            print()
        fold = f"fold_{i+1}"
        df[fold] = 'train'
        df.loc[val_index, fold] = 'vali'
        df_train_split = df[df[f"fold_{i+1}"]=='train']
        df_vali = df[df[f"fold_{i+1}"]=='vali']
        df_val_split, df_test_split = stratified_test_split(df_vali, n_splits=2, stratify_by=stratify_by, group='patient_id', seed=None)
        df_val_split.loc[:, fold] = 'vali'
        df_test_split.loc[:, fold] = 'test'
        df = pd.concat([df_train_split, df_val_split, df_test_split ])

    return df

def stratified_test_split(df, n_splits=2, stratify_by='days_to_birth', group='patient_id', seed=None):
    if seed is None: 
        shuffle = False 
    else:
        shuffle = True
    y = df[stratify_by]
    X = df.drop(stratify_by, axis=1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
    split = splitter.split(X=X,y=y,groups=df[group])
    part1_idx, part2_ids = next(split)
    df1 = df.iloc[part1_idx]
    df2 = df.iloc[part2_ids]
    return df1, df2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create train/val/test splits with non-overlapping patients")
    parser.add_argument("--seed", type=int, default=20)    
    parser.add_argument("--stratify_by", type=str, default='ga_in_weeks')
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument('--csv_preterm', type=str, help='path to csv for preterm data', default='/home/ppar/SA-SonoNet-sPTB/metadata/dataset_preterm.csv')
    parser.add_argument('--csv_term', type=str, help='path to csv for term data', default='/home/ppar/SA-SonoNet-sPTB/metadata/dataset_term.csv')
    args = parser.parse_args()
    
    df_pre = pd.read_csv(args.csv_preterm)
    df_term = pd.read_csv(args.csv_term)
    
    print()
    print('Full image dataset')
    print(50*'-')
    print(f'Unique patients | preterm: {df_pre.patient_id.value_counts().shape[0]}, term: {df_term.patient_id.value_counts().shape[0]}')
    print(f'Numeber of images | preterm: {df_pre.shape[0]}, term:  {df_term.shape[0]}')
    print(50*'-')
    print()

    # make sure we remove duplicates (this was actually done as preprocessing step)
    df_pre = df_pre.drop_duplicates(subset='image_dir_clean', keep=False)
    df_term = df_term.drop_duplicates(subset='image_dir_clean', keep=False)
  
    # get clinical prediction standard based on cervical length
    df_pre['pred'] = np.where(df_pre['cervical_length'] < 25, 1, 0)
    df_term['pred'] = np.where(df_term['cervical_length'] < 25, 1, 0)
    
    # drop rows based on constrains
    df_pre = drop_rows(df_pre, term=False)
    df_term = drop_rows(df_term, term=True)

    # keep one image per patient for term birhts
    df_term = df_term.drop_duplicates(subset=['patient_id', 'ga', 'date_of_birth'], keep='last')

    print('Sampled image dataset')
    print(50*'-')
    print(f'Unique patients | preterm: {df_pre.patient_id.value_counts().shape[0]}, term: {df_term.patient_id.value_counts().shape[0]}')
    print(f'Numeber of images | preterm: {df_pre.shape[0]}, term:  {df_term.shape[0]}')
    print(50*'-')
    print()

    # sample term images: same number of images per GA as in preterm birhts
    df_pre_group_counts = df_pre.groupby('ga_in_weeks').size()
    df_term_data = df_term.groupby('ga_in_weeks').apply(lambda x: x.sample(df_pre_group_counts[x.name], random_state=args.seed, replace=0))

    print('Final image dataset')
    print(50*'-')
    print(f'Unique patients | preterm: {df_pre.patient_id.value_counts().shape[0]}, term: {df_term_data.patient_id.value_counts().shape[0]}')
    print(f'Numeber of images | preterm: {df_pre.shape[0]}, term:  {df_term_data.shape[0]}')
    print(50*'-')
    print()

    # merge preterm and term data
    df_term_data = df_term_data.reset_index(drop=True)
    df = pd.concat([df_pre, df_term_data], axis=0)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print('Number of images in the final dataset: ', df.shape[0])
    print()

    # create splits
    df = stratified_split_patients(df, n_splits=args.n_splits, stratify_by=args.stratify_by, group='patient_id', seed=None)
    df = df.reset_index(drop=True)

    # swap val and test data in each split
    new_df = df.copy()
    fold_columns = [f'fold_{fold_index}' for fold_index in range(1, args.n_splits+1)]
    new_fold_columns = [f'fold_{fold_index}B' for fold_index in range(1, args.n_splits+1)]
    for fold_col, new_fold_col in zip(fold_columns, new_fold_columns):
        new_df[new_fold_col] = df[fold_col].replace({'vali':'test', 'test':'vali'})
    all_fold_cols = fold_columns + new_fold_columns
    df = new_df.copy()

    # save
    if args.save:
        updated_csv_name = args.csv_preterm.rsplit('/', 1)[0] + '/ASMUS_MICCAI_dataset_splits' + '.csv'
        df.to_csv(f'{updated_csv_name}', index=False)

    if args.eval:
        print('\nResults: Clinical Standard - Cervical Length\n')
        auc_list, acc_list, recall_list, specificity_list = [],[],[], [] 
        for index, fold_id in enumerate(all_fold_cols):
            df_train = df[df[fold_id]=='train']
            df_val = df[df[fold_id]=='vali']
            df_test = df[df[fold_id]=='test']
            df_train = df_train.reset_index(drop=True)
            df_val = df_val.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)

            df_final = pd.concat([df_train, df_val, df_test])
            df_final = df_final.reset_index(drop=True)
            df_final = df_final.sample(frac=1, random_state=args.seed).reset_index(drop=True)

            df_term_final = df_final[df_final.birth_before_week_37!=1].reset_index(drop=True)
            df_preterm_final = df_final[df_final.birth_before_week_37==1].reset_index(drop=True)
            df_term_train = df_term_final[df_term_final[fold_id] == 'train'].reset_index(drop=True)
            df_term_val = df_term_final[df_term_final[fold_id]== 'vali'].reset_index(drop=True)
            df_term_test = df_term_final[df_term_final[fold_id]== 'test'].reset_index(drop=True)
            df_pre_term_train = df_preterm_final[df_preterm_final[fold_id] == 'train'].reset_index(drop=True)
            df_pre_term_val = df_preterm_final[df_preterm_final[fold_id] == 'vali'].reset_index(drop=True)
            df_pre_term_test = df_preterm_final[df_preterm_final[fold_id] == 'test'].reset_index(drop=True)

            auc, acc, recall, specificity = cervical_length_results(df_test)
            print(f'fold: {fold_id},  AUC: {np.round(auc, 3)}, Acc: {np.round(acc, 3)}, Recall: {np.round(recall, 3)}, Specificity: {np.round(specificity, 3)}')
            auc_list.append(auc)
            acc_list.append(acc)
            recall_list.append(recall)
            specificity_list.append(specificity)

        mean_acc, std_acc = np.round(np.mean(acc_list), 3), np.round(np.std(acc_list), 3)
        mean_auc, std_auc = np.round(np.mean(auc_list), 3), np.round(np.std(auc_list), 3)
        mean_recall, std_recall = np.round(np.mean(recall_list), 3), np.round(np.std(recall_list), 3)
        mean_specificity, std_specificity = np.round(np.mean(specificity_list), 3), np.round(np.std(specificity_list), 3)
        print()
        print(f'AUC: {mean_auc}±{std_auc}, Acc: {mean_acc}±{std_acc}, Recall: {mean_recall}±{std_recall}, Specificity: {mean_specificity}±{std_specificity}')