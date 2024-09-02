import ast
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
import torch
from torchvision import transforms as T

def pca_textural_features(df, feature_names, fold_index, n_components=32):

    # get training split
    df_train = df[df[fold_index]=='train']

    # standar scaler using mean and std of training split
    for feature in feature_names:
        mean, std = np.mean(df_train[feature]), np.std(df_train[feature])
        df[f'{feature}'] = (df[feature] - mean) / std
    df_train_scaled = df[df[fold_index]=='train']
    
    # fit pca features
    pca = PCA(n_components=n_components)
    pca.fit(df_train_scaled[feature_names])
    
    # combine metadata and pca features datadframes
    df_metadata = df[df.columns[~df.columns.isin(feature_names)]]
    df_pca_features = pd.DataFrame(pca.transform(df[feature_names]))
    df_combined = pd.concat([df_metadata, df_pca_features], axis=1)

    return df_combined

class PretermBirthTextureDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        split:str = 'train', 
        csv_dir:str = '/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_texture_dataset_splits.csv',
        feature_names_dir:str = '/home/ppar/SA-SonoNet-sPTB/metadata/textural_features.txt',
        split_index:str = 'fold_1',
        label_name= 'birth_before_week_37',
        n_components:int=32,
        class_only:int = -1,
        **kwargs,
    ):
        super().__init__()

        assert split in ['train', 'vali', 'test', 'valitest','all']
        
        
        csv = pd.read_csv(csv_dir)

        with open(feature_names_dir, 'r') as f:
            feature_names = ast.literal_eval(f.read())

        csv = pca_textural_features(
            csv,
            feature_names=feature_names, 
            fold_index=split_index, 
            n_components=n_components
        )        

        if split =='train':
            csv = csv[csv[split_index]=='train']
        elif split =='vali':
            csv = csv[csv[split_index]=='vali']
        elif split =='test':
            csv = csv[csv[split_index]=='test']
        elif split == 'valitest':
            vali_csv = csv[csv[split_index]=='vali']
            test_csv = csv[csv[split_index]=='test']
            csv = pd.concat((vali_csv, test_csv), axis=0)
            csv = pd.concat((vali_csv, test_csv, csv), axis=0)
        elif split == 'all':
            pass
        else:
            csv = csv[csv[split_index]==split]

        self.split = split
        self.split_index = split_index
        self.feature_names = feature_names
        self.label_name = label_name
        self.n_components = n_components
        self.train_csv = csv[csv[split_index]=='train']

        if class_only!=-1:
            csv = csv[csv[label_name] == class_only]
        
        self.csv = csv.reset_index(drop=True)
    
    def __len__(self):
        return len(self.csv)
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):
        label = int(self._get_attr(index, self.label_name))

        textural_features = self._get_attr(index, list(range(0,self.n_components))).values
        textural_features = torch.from_numpy(textural_features.astype(np.float32))
        # wrapup
        data = {}
        data['texture_features'] = textural_features
        data['label'] = label
        return data
    
    def _get_label(self, index):
        label_name = int(self._get_attr(index, self.label_name))
        return label_name
    
    def get_labels(self):
        labels = np.array([self._get_label(l) for l in range(len(self.csv))])
        return labels

if __name__ == "__main__":
    split_index='fold_5'
    traindata = PretermBirthTextureDataset(split='train', split_index=split_index)
    print(f"training data: {len(traindata)}")
    valdata = PretermBirthTextureDataset(split='vali', split_index=split_index)
    print(f"validation data: {len(valdata)}")
    testdata = PretermBirthTextureDataset(split='test', split_index=split_index)
    print(f"test data: {len(testdata)}")
    loader = torch.utils.data.DataLoader(valdata, batch_size=64,shuffle=False)

    for i, data in enumerate(loader):
        print(data['texture_features'].shape, data['label'].shape)

