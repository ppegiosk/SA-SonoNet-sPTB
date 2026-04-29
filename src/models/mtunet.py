import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, ConvBlock, DeConvBlock

class MTUNet(nn.Module):
    """
    Unofficial implementation of the multi-task UNet proposed by Włodarczy et. al. [1]
    [1] Spontaneous preterm birth prediction using convolutional neural networks 
    """
    def __init__(
        self, 
        in_channels=1, seg_labels=1, cls_labels=1, 
        channels=[32, 64, 128, 256, 512]
    ):
        super().__init__()

        self.channels = channels
        self.down_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.down_convs.append(
            ConvBlock(
                in_channels=in_channels, 
                out_channels=channels[0], 
                stride=1)
        )

        num_layers = len(channels) - 1
        for i in range(num_layers):
            self.down_convs.append(
                ConvBlock(
                    in_channels=channels[i], 
                    out_channels=channels[i+1], 
                    stride=2)
            )
            self.up_convs.append(
                DeConvBlock(
                    left_channels=channels[-(i+2)], 
                    right_channels = channels[-(i+1)], 
                    out_channels=channels[-(i+2)])
            )

        self.up_convs.append(
        DeConvBlock(
            left_channels=channels[0], 
            right_channels=channels[1], 
            out_channels=channels[0])
        )

        self.last_conv = nn.Conv2d(channels[0], seg_labels, kernel_size=1)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(channels[-1] * 2, cls_labels)

    def forward(self, x):

        x = x['image']

        downs = []
        for down_conv in self.down_convs:
            x = down_conv(x)
            downs.append(x)
        downs = downs[:-1]
        downs = reversed(downs)
        for i, (down, up_conv) in enumerate(zip(downs, self.up_convs)):
            if i==0:
                classification_input = x.clone()
            x = up_conv(down, x)

        classification_input = self.dropout(self.flatten(classification_input))
        classification_logit = self.classifier(classification_input)
        segmentation_logit = self.last_conv(x)

        outputs = {}
        outputs['logit'] = classification_logit
        outputs['seg_logit'] = segmentation_logit
        return outputs




if __name__ == "__main__":
    model = MTUNet(1, 1, 1)
    image = torch.rand(64, 1, 224, 288)
    outputs = model({'image': image})
    print(outputs['seg_logit'].shape)
    print(outputs['logit'].shape)
    