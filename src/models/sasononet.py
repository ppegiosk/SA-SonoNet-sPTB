"""
Official PyTorch implementation of SA-SonoNet
Paraskevas Pegios et al., Leveraging Shape and Spatial Information for Spontaneous Preterm Birth Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.sononet import SonoNet
from src.models.dtunet import DTUNet

FEAT_CHANNELS = {'SN16': 128, 'SN32': 256,  'SN64': 512}
FRIST_CONV = {'SN16': 16, 'SN32': 32,  'SN64': 64}

def sononet_extractor(config, in_channels=8, num_labels=1):

    channels = int(FEAT_CHANNELS[config])
    first_conv_channels = int(FRIST_CONV[config])

    # pretrained sononet
    model = SonoNet(config=config,
                    num_labels=num_labels,
                    features_only=True, 
                    weights=True,
                    in_channels=1 # pretrained on grayscale images 
            )

    # Modification of the first conv layer to match input channnel size
    if in_channels !=1:
        model.features[0][0][0] = nn.Conv2d(in_channels, first_conv_channels, 
                                            kernel_size=3, padding=1, bias=False
        )
        for name, param in model.named_parameters():
            if name=='features.0.0.0.weight' or name=='features.0.0.0.bias':
                param.requires_grad = True

    return model

class SASonoNetBase(nn.Module):
    def __init__(self, config, num_labels, in_channels=8):
        super(SASonoNetBase, self).__init__()

        channels = int(FEAT_CHANNELS[config])
        self.feature_extractor = sononet_extractor(config=config, 
                                 in_channels=in_channels, 
                                 num_labels=num_labels
        )
        self.adaption = nn.Sequential(
                         nn.Conv2d(channels,
                                   channels // 2, 1, bias=False),
                         nn.BatchNorm2d(channels // 2),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(channels // 2, num_labels, 1, bias=False),
                         nn.BatchNorm2d(num_labels),
        )

    def forward(self, x):
        # feature extractor
        features = self.feature_extractor(x)
        # adaptation layer
        adapt = self.adaption(features)
        # classification layer
        logit = F.avg_pool2d(adapt, adapt.size()[2:]).view(adapt.size(0), -1)
        return logit, features

class SASonoNet(SASonoNetBase):
    def __init__(self, config, num_labels, in_channels=8):
        super().__init__(config, num_labels, in_channels)
        
        assert in_channels in [1, 3, 8]
        self.in_channels = in_channels

    def forward(self, inputs):
        if self.in_channels == 1:
            x = inputs['image']
        elif self.in_channels == 3:
            x = inputs['image_spacing']
        elif self.in_channels == 8:
            x = inputs['image_segpred_spacing']
        logit, features = super().forward(x=x)
        return {'logit': logit, 
                'features': features
        }

class SASonoNetModel(nn.Module):
    """
    Official PyTorch implementation of SA-SonoNet
    Paraskevas Pegios et al., 
    Leveraging Shape and Spatial Information for Spontaneous Preterm Birth Prediction
    """
    def __init__(
        self, 
        config='SN32', num_labels=1, in_channels=8,
        dtunet_checkpoint='/home/ppar/SA-SonoNet-sPTB/src/models/weights/dtunet/model.t7',
        device='cpu',
    ):
        super(SASonoNetModel, self).__init__()

        # DTU-Net trained on an external multi-class segmentation dataset with L = 14 structures
        dtunet = DTUNet(in_channels=1, out_channels=14).to(torch.device(device))
        # load DTU-Net weights
        dtunet.load_state_dict(torch.load(dtunet_checkpoint,  map_location=torch.device(device)))
        # fix trained DTU-Net
        for name, param in dtunet.named_parameters():
            param.requires_grad = False

        self.dtunet_checkpoint = dtunet_checkpoint      
        self.dtunet = dtunet
        self.num_labels = num_labels

        # initialize feature extractor SA-SonoNet with pre-trained SonoNet weights
        channels = int(FEAT_CHANNELS[config])
        self.feature_extractor = sononet_extractor(config=config, 
                                 in_channels=in_channels, 
                                 num_labels=num_labels
        )

        # adapation layer for the classification task
        self.adaption = nn.Sequential(
                         nn.Conv2d(channels,
                                   channels // 2, 1, bias=False),
                         nn.BatchNorm2d(channels // 2),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(channels // 2, num_labels, 1, bias=False),
                         nn.BatchNorm2d(num_labels),
        )

    def forward(self, inputs):

        self.dtunet.eval()

        # raw input image
        image_with_calipers = inputs['image_with_calipers']
        # input image without confounders (text & calipers)
        image = inputs['image']
        # spatial information: pixel spacing 
        spacing = inputs['spacing']

        # shape information: segmentation predictions 
        with torch.no_grad():
            segmentation_logits = self.dtunet({'image':image_with_calipers})['logit']
            # cervical shape information: keep only segmentations relevant for the task (K = 5)
            segmentation_logits = segmentation_logits[:, :5, : ,: ].detach()
        
        # combine image, shape and spatial information
        x = torch.cat((image, segmentation_logits, spacing), dim=1)
        
        # feature extractor
        features = self.feature_extractor(x)
        # adaptation layer
        adapt = self.adaption(features)
        # classification layer
        logit = F.avg_pool2d(adapt, adapt.size()[2:]).view(adapt.size(0), -1)
        
        # output risk score, features, and DTU-Net segmentation mask (used for CL estimates and feedback)
        return {'logit': logit, 
                'features': features,
                'mask': segmentation_logits,
        }


if __name__ == '__main__':
    import numpy as np
    image = torch.rand(64, 1, 224, 288)
    image_spacing = torch.rand(64, 3, 224, 288)
    image_segpred_spacing = torch.rand(64, 8, 224, 288)

    model = SASonoNet(config='SN32',
                      in_channels=8, 
                      num_labels=1)
    #print(model)
    output = model({'image':image, 
                   'image_spacing': image_spacing, 
                   'image_segpred_spacing': image_segpred_spacing})
    logit = output['logit']
    print(f'Model output shape: {logit.shape}')

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in parameters])
    print(f'Number of parameters: {params}')

