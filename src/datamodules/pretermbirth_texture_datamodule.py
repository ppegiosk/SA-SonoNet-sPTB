import numpy as np
from torchvision import transforms as T
import pandas as pd
from PIL import Image
import torch
import yaml
import pytorch_lightning as pl
from torchsampler import ImbalancedDatasetSampler
import albumentations as A

from src.datamodules.pretermbirth_texture_dataset import PretermBirthTextureDataset


class PretermBirthTextureDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int=64,
        data='splits',
        split_index='fold_1',
        label='birth_before_week_37',
        **kwargs 
    ):
        
        super().__init__()

        self.data = data
        self.split_index = split_index
        self.label_name = label
        self.batch_size = batch_size
        self.num_workers = 16

        if self.data == 'splits':
            self.csv_dir = '/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_texture_dataset_splits.csv'
        
        
    def train_dataloader(self):

        trainset = PretermBirthTextureDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index,
            label_name=self.label_name, 
            split='train'
        )

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return trainloader
    
    def val_dataloader(self, shuffle=False):
        valset = PretermBirthTextureDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index,
            label_name=self.label_name,
            split='vali'
        )
        
        valloader = torch.utils.data.DataLoader(
            valset, batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )
        return valloader
    
    def test_dataloader(self):

        testset = PretermBirthTextureDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index,
            label_name=self.label_name, 
            split='test')

        testloader = torch.utils.data.DataLoader(
            testset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )

        return testloader    


if __name__ == "__main__":
    import albumentations as A
    from matplotlib import pyplot as plt
    import numpy as np
    batchsize = 1
    dm = PretermBirthTextureDataModule(batch_size=batchsize, data='splits', split_index='fold_1B')

    print(dm.train_dataloader().dataset.__len__())
    print(dm.val_dataloader().dataset.__len__())
    print(dm.test_dataloader().dataset.__len__())
    print(dm.valtest_dataloader().dataset.__len__())