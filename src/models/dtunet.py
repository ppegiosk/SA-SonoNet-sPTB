# Adapted from https://github.com/mmmmimic/DTU-Net/blob/main/models/unet_family.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, ConvBlock, DeConvBlock, Conv3x3

class DTUNet(nn.Module):
    def __init__(self, in_channels, out_channels,  pretrained=False):
        super().__init__()
        self.unet = RegUNet(in_channels, num_classes=out_channels, pretrained=False)
        # toponet
        self.left_conv1 = ConvBlock(out_channels, 64)
        self.left_conv2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(64, 128)
            )
        self.left_conv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(128, 256)
            )
        self.left_conv4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(256, 512)
            )        

        self.right_conv4 = DeConvBlock(512, 256, 256)
        self.right_conv3 = DeConvBlock(256, 128, 128)
        self.right_conv2 = DeConvBlock(128, 64, 64)
        self.last_conv = nn.Conv2d(64, 1, kernel_size=1, 
                                   padding=0, stride=1)             

        self.class_num = out_channels

        self.attention_module = nn.Sequential(
                        Conv3x3(in_channels+out_channels+1, 32),
                        Conv3x3(32, 32),
                        Conv3x3(32, 2)
        )

    @staticmethod
    def random_erase(img):
        h, w = img.shape
        if not img.max():
            return 1 - img
        
        kernel_size = 8
        vals = torch.max_pool2d((img>0).float().view(1, 1, h, w), kernel_size=kernel_size, stride=kernel_size).squeeze()
        vals = vals.view(-1)
        idx = torch.nonzero(vals)

        patches = img.unfold(0, kernel_size, kernel_size).unfold(1, kernel_size, kernel_size)
        
        m, n, _, _ = patches.shape
        patches = patches.contiguous().view(-1, kernel_size, kernel_size)

        num = len(idx)
        index = np.array(range(num))
        np.random.shuffle(index)
        sample_index = idx[index[:int(num*0.2)]]
        patches[sample_index, ...] = 0

        patches = patches.view(m, n, kernel_size, kernel_size)
        patches = patches.contiguous()

        img = patches.permute(0,2,1,3).contiguous()
        img = img.view(h, w)

        return img

    def forward_once(self, x):
        l1 = self.left_conv1(x)
        l2 = self.left_conv2(l1)
        l3 = self.left_conv3(l2)
        l4 = self.left_conv4(l3)
        return l1, l2, l3, l4

    def forward(self, x):
        # coarse segmentation
        unet_out = self.unet(x) # [B, C, H, W]
        coarse_logit = unet_out['logit']
        emb = unet_out['emb']

        coarse_score = torch.softmax(coarse_logit, dim=1)

        anchor_features = self.forward_once(coarse_score)

        if self.training: # during training, activate triplet loss
            mask = x['mask']
            crp_mask = torch.zeros_like(mask)
            for i in range(mask.size(0)):
                crp_mask[i,...] = self.random_erase(mask[i,...].clone())
            mask = F.one_hot(mask.long(), self.class_num).permute(0,3,1,2).float()
            crp_mask = F.one_hot(crp_mask.long(), self.class_num).permute(0,3,1,2).float()

            mask_features = self.forward_once(mask)[-1]
            crp_mask_features = self.forward_once(crp_mask)[-1]

            triplet_loss = nn.TripletMarginLoss(margin=0.1)(anchor_features[-1], 
                            mask_features, crp_mask_features)
        else:
            triplet_loss = torch.tensor([0]).to(coarse_score.device)

        triplet_loss = torch.tensor([0]).to(coarse_score.device)

        l1, l2, l3, l4 = anchor_features
        r3 = self.right_conv4(l3, l4)
        r2 = self.right_conv3(l2, r3)
        r1 = self.right_conv2(l1, r2)
        bf_mask = torch.sigmoid(self.last_conv(r1)) 

        if bf_mask.size(-1) != coarse_score.size(-1):
            bf_mask = F.interpolate(bf_mask, (coarse_score.size(-2), 
                                coarse_score.size(-1)))

        weight = self.attention_module(torch.cat((coarse_score, bf_mask, x['image']), dim=1))
        weight = torch.softmax(weight, dim=1)
        tex_weight, topo_weight = torch.split(weight, [1,1], dim=1)
        bg, fg = torch.split(coarse_score, [1, coarse_score.size(1)-1], dim=1)
        bg, fg = (tex_weight*bg+topo_weight*(1-bf_mask)), tex_weight*fg+topo_weight*bf_mask
        logit = torch.cat((bg, fg), dim=1)   
        logit = torch.nn.functional.normalize(logit, dim=1, p=1)

        out = {}
        out['coarse_logit'] = coarse_logit
        out['logit'] = logit
        out['topo_mask'] = bf_mask.squeeze(1)
        out['triplet_loss'] = triplet_loss
        out['emb'] = emb

        return out


class RegUNet(nn.Module):
    def __init__(self, in_channels, num_classes, model_size = '016', model_type= 'y', interpolation=True, pretrained=True, **kwargs):
        super().__init__()

        if model_size == '004':
            channels = [32,48,104,208,440]
        elif model_size == '008':
            channels = [32,64,128,320,768]
        elif model_size == '016':
            channels = [32,48,120,336,888]
        elif model_size == '032':
            channels = [32,72,216,576,1512]
        else:
            raise Exception(f"Invalid model_size '{model_size}'")
        
        # get regnet encoder
        self.down_conv0, self.down_conv1, self.down_conv2, self.down_conv3, self.down_conv4 = get_regnet(model_size=model_size, model_type=model_type, in_channels=in_channels, pretrained=pretrained)

        self.up_conv4 = DeConvBlock(channels[-2], channels[-1], out_channels=channels[-2], **kwargs)

        self.up_conv3 = DeConvBlock(channels[-3], channels[-2], out_channels=channels[-3], **kwargs)

        self.up_conv2 = DeConvBlock(channels[-4], channels[-3], out_channels=channels[-4], **kwargs)

        self.up_conv1 = DeConvBlock(channels[-5], channels[-4], out_channels=channels[-5], **kwargs)

        self.up_conv0 = DeConvBlock(0, channels[-5], out_channels=num_classes, **kwargs)


    def forward(self, x):
        x = x['image']
        # left
        down1 = self.down_conv0(x)
        down2 = self.down_conv1(down1)
        down3 = self.down_conv2(down2)
        down4 = self.down_conv3(down3)

        # bottleneck
        down5 = self.down_conv4(down4)
        emb = torch.mean(down5.flatten(2), dim=-1)

        # right
        up4 = self.up_conv4(down4, down5)
        up3 = self.up_conv3(down3, up4)
        up2 = self.up_conv2(down2, up3)
        up1 = self.up_conv1(down1, up2)
        
        out = self.up_conv0(None, up1)

        out_dict = {}
        out_dict['logit'] = out 
        out_dict['emb'] = up1

        return out_dict


def get_regnet(in_channels=3,
                embedding_size=1024,
                model_size = '016',
                model_type = 'y',
                pretrained=True):

    import timm.models as models
    assert model_size in ['004', '008', '016', '032']
    assert model_type in ['x', 'y']
    
    model = eval("models.regnet.regnet%s_%s(pretrained=pretrained, num_classes=embedding_size)"%(model_type, model_size))

    if in_channels != 3:
        model.stem.conv = torch.nn.Conv2d(in_channels=in_channels,
                                            out_channels=model.stem.conv.out_channels,
                                            kernel_size=model.stem.conv.kernel_size,
                                            stride = model.stem.conv.stride,
                                            padding=model.stem.conv.padding,
                                            bias = model.stem.conv.bias)

    return model.stem, model.s1, model.s2, model.s3, model.s4