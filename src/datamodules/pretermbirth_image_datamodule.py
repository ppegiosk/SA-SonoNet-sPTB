import albumentations as A
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from src.datamodules.pretermbirth_image_dataset import PretermBirthImageDataset

class PretermBirthImageDatamdule(pl.LightningDataModule):
    def __init__(
        self, 
        data:str ='splits',
        split_index:str = 'fold_1',
        label:str = 'birth_before_week_37',
        batch_size: int = 64,
        img_size:tuple = (224, 288), 
        **kwargs 
    ):
        
        super().__init__()

        self.data = data
        self.split_index = split_index
        self.label_name = label
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = 16

        if 'ablation_id' in list(kwargs.keys()):
            self.ablation_id = kwargs['ablation_id']
            self.class_only = kwargs['class_only']

        if self.data == 'splits':
            self.csv_dir = '/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_dataset_splits.csv'
        elif self.data == 'ablation':
            self.csv_dir = '/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_dataset_splits.csv'
        elif self.data == 'external':
            self.csv_dir = '/home/ppar/SA-SonoNet-sPTB/metadata/external_testset_all.csv'
        
        
    def train_dataloader(self):

        train_transforms = [
            A.CLAHE(p=0.2),
            A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5), # TODO: add vertical flip augmentation
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=[-25, 25], p=0.5, border_mode=0),
            A.RandomGamma(p=0.5),
            A.Resize(self.img_size[0], self.img_size[1])
          ]

        trainset = PretermBirthImageDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index, 
            transforms=train_transforms, 
            label_name=self.label_name, 
            split='train')

        trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True
        )
        return trainloader
    
    def val_dataloader(self, shuffle=False):

        val_transforms = [A.Resize(self.img_size[0], self.img_size[1])]

        valset = PretermBirthImageDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index, 
            transforms=val_transforms, 
            label_name=self.label_name,
            split='vali'
        )
        
        valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        return valloader
    
    def test_dataloader(self):

        test_transforms = [A.Resize(self.img_size[0], self.img_size[1])]

        testset = PretermBirthImageDataset(
            csv_dir=self.csv_dir, split_index=self.split_index, 
            transforms=test_transforms,
            label_name=self.label_name,
            split='test'
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )

        return testloader
    
    def valtest_dataloader(self):

        test_transforms = [A.Resize(self.img_size[0], self.img_size[1])]

        testset = PretermBirthImageDataset(
            csv_dir=self.csv_dir, 
            split_index=self.split_index, 
            transforms=test_transforms, 
            label_name=self.label_name,
            # class_only=self.class_only,
            split='valtest'
        )

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size,
            shuffle=False, 
            num_workers=self.num_workers
        )

        return testloader
    
if __name__ == "__main__":
    dm = PretermBirthImageDatamdule(batch_size=1, data='splits', split_index='fold_1B')
    print(dm.train_dataloader().dataset.__len__())
    print(dm.val_dataloader().dataset.__len__())
    print(dm.test_dataloader().dataset.__len__())