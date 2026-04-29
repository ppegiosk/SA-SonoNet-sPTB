import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  

class TextureNet(nn.Module):
    """
    MLP with hand-crafted textural features extracted from a ROI around the cervical canal
    Inspired by [1,2,3]
    [1] Baños, N., et al.: Quantitative analysis of cervical texture by ultrasound in mid-pregnancy and association with spontaneous preterm birth
    [2] Bustamante, D., et al.: Cervix ultrasound texture analysis to differentiate between term and preterm birth pregnancy: a machine learning approach
    [3] Fiset, S., et al.: Prediction of spontaneous preterm birth among twin gestations using machine learning and texture analysis of cervical ultrasound images
    """
    def __init__(self, in_channels=32, num_labels=1, fc_dim=128):
        super().__init__()

        self.texture_classifier = nn.Sequential(
            nn.Linear(in_channels, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(fc_dim, fc_dim//2),
            nn.BatchNorm1d(fc_dim//2),
            nn.LeakyReLU(), 
            nn.Dropout(0.5),
            nn.Linear(fc_dim//2, num_labels),
        )
        
    def forward(self, data):
        x = data['texture_features']
        logit = self.texture_classifier(x)
        output = {}
        output['logit'] = logit
        return output


if __name__ == "__main__":
    model = TextureNet(in_channels=32, num_labels=1, fc_dim=128)
    texture_features = torch.rand(64, 32)
    output = model({'texture_features': texture_features})
    print(output['logit'].shape)
    