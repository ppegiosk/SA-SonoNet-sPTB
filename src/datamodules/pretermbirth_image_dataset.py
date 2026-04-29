import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms as T
import torch

class PretermBirthDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self, 
        split:str = 'train', 
        csv_dir:str = '/home/ppar/SA-SonoNet-sPTB/metadata/ASMUS_MICCAI_dataset_splits.csv',
        split_index:str = 'fold_1',
        **kwargs,
    ):
        super().__init__()

        assert split in ['train', 'vali', 'test', 'valitest', 'all']
        
        # read file name list
        csv = pd.read_csv(csv_dir, low_memory=False)

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

        self.csv = csv.reset_index(drop=True)
        self.split = split
        self.split_index = split_index
    
    def _get_attr(self, index, attr):
        return self.csv.loc[index, attr]

    def __getitem__(self, index):
        # read image
        image_wo_calipers_dir = self._get_attr(index, 'image_dir_clean')
        image = Image.open(image_wo_calipers_dir)
        image = np.asarray(image)
        # read image
        image_w_calipers_dir = self._get_attr(index, 'image_dir_calipers')
        image_with_calipers = Image.open(image_w_calipers_dir)
        image_with_calipers = np.asarray(image_with_calipers)
        # read mask
        mask_dir = self._get_attr(index, 'mask_dir')
        segmentation_data = np.load(mask_dir, allow_pickle=True).item()
        segmentation_mask = segmentation_data['zahra']
        segmentation_logits = segmentation_data['logits']

        # breakpoint()
        binary_mask = segmentation_data['binary']
        segmentation_logits = segmentation_logits.transpose((1,2,0))
        segmentation_mask[segmentation_mask>4] = 0

        return image, image_with_calipers, segmentation_mask, binary_mask, segmentation_logits

    def __len__(self):
        return len(self.csv)

class PretermBirthImageDataset(PretermBirthDatasetBase):
    def __init__(self, 
                 transforms, 
                 label_name:str = 'birth_before_week_37', 
                 class_only:int = -1,
                 **kwargs           
    ):
        super().__init__(**kwargs)

        assert class_only in [-1, 0, 1] # if 0 get only term if 1 get only preterm

        if class_only!=-1:
            self.csv = self.csv [self.csv[label_name] == class_only]
            self.csv = self.csv.reset_index(drop=True)

        self.transforms = transforms
        self.label_name = label_name

    def __getitem__(self, index):

        image, image_with_calipers, segmentation_mask, binary_mask, segmentation_logits = super().__getitem__(index)

        # resize images
        tf_resize = [self.transforms[-1]] if not isinstance(self.transforms[-1], list) \
            else self.transforms[-1]

        resize = A.Compose(
            tf_resize, 
            additional_targets={
                'image_with_calipers': 'image',
                'binary_mask':'mask', 
                'segmentation_logits': 'mask' 
                }
            )
        data = resize(
            image=image,
            image_with_calipers=image_with_calipers,
            mask=segmentation_mask, 
            binary_mask=binary_mask, 
            segmentation_logits=segmentation_logits)

        image, image_with_calipers, segmentation_mask, binary_mask, segmentation_logits = \
            data['image'], data['image_with_calipers'], data['mask'], data['binary_mask'], data['segmentation_logits']

        # get pixel spacing for resized images from metadata csv file
        px_spacing_resized = float(self._get_attr(index, 'px_spacing')) 
        py_spacing_resized = float(self._get_attr(index, 'py_spacing'))

        # data augmentation
        tf_augmentation = A.Compose(
            self.transforms, 
            additional_targets={
                'image_with_calipers': 'image',
                'binary_mask':'mask', 
                'segmentation_logits': 'mask'
                }
        )

        aug_data = tf_augmentation(
            image=image,
            image_with_calipers=image_with_calipers, 
            mask=segmentation_mask, 
            binary_mask=binary_mask, 
            segmentation_logits=segmentation_logits
        )

        image, image_with_calipers, segmentation_mask, binary_mask, segmentation_logits = \
            aug_data['image'], aug_data['image_with_calipers'], aug_data['mask'], aug_data['binary_mask'], aug_data['segmentation_logits']        
        
        binary_mask = T.ToTensor()(np.float32(binary_mask))
        segmentation_mask = T.ToTensor()(np.float32(segmentation_mask))
        segmentation_logits = T.ToTensor()(np.float32(segmentation_logits))

        # normlize images in [-1, 1] and covert to grayscale
        image = T.ToTensor()(np.array(image))
        image = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)
        image = torch.mean(image, axis=0, keepdims=True)

        image_with_calipers = T.ToTensor()(np.array(image_with_calipers))
        image_with_calipers = T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image_with_calipers)
        image_with_calipers = torch.mean(image_with_calipers, axis=0, keepdims=True)

        # get label from metadata csv file
        label = float(self._get_attr(index, f'{self.label_name}'))

        # repeat, reshape pixel spacing info and concatenate input channels
        pixelspacing_x = np.empty(image.shape)
        pixelspacing_x.fill(px_spacing_resized * 10)
        pixelspaxing_y = np.empty(image.shape)
        pixelspaxing_y.fill(py_spacing_resized * 10)
        pixel_spacing = np.concatenate([pixelspacing_x, pixelspaxing_y])
        pixel_spacing = torch.from_numpy(np.float32(pixel_spacing))

        # wrapup
        data = {}
        data['image_spacing'] = torch.concat([image, pixel_spacing], axis=0)
        data['spacing'] = pixel_spacing
        data['image_segpred_spacing'] = torch.concat([image, segmentation_logits, pixel_spacing], axis=0)
        data['label'] = label

        data['image'] = image
        data['image_with_calipers'] = image_with_calipers
        data['binary_mask'] = binary_mask
        data['segmentation_mask'] = segmentation_mask
        data['segmentation_logits'] = segmentation_logits
        return data
    
    def _get_label(self, index):
        label = int(self._get_attr(index, self.label_name))
        return label
    
    def get_labels(self):
        labels = np.array([self._get_label(l) for l in range(len(self.csv))])
        return labels