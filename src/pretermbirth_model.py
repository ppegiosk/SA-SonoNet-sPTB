import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch import nn
from torchmetrics.functional import auroc, recall, precision, specificity, accuracy

from src.models.texturenet import TextureNet
from src.models.sasononet import SASonoNet, SASonoNetModel
from src.models.mtunet import MTUNet

class PretermBirthModel(pl.LightningModule):
    def __init__(self, hparams):

        super().__init__()

        self.save_hyperparameters(hparams)

        assert self.hparams.model.lower() \
            in ['sa-sononet-16', 'sa-sononet-32', 'sa-sononet-64', 'mt-unet', 'texturenet']

        if 'sa-sononet' in self.hparams.model.lower():
            config = 'SN' + self.hparams.model.lower().split('sa-sononet-')[-1]
            self.net = SASonoNet(
                config=config, 
                in_channels=self.hparams.in_channels, # img + seg pred + spacing channels
                num_labels=1,
            )

            # classification loss
            self.bce_loss = torch.nn.BCEWithLogitsLoss()

        elif 'mt-unet' in self.hparams.model.lower():
            self.hparams.in_channels = 1 # grayscale img
            self.net = MTUNet(
                in_channels=self.hparams.in_channels,
                cls_labels=1, 
                seg_labels=1,
            )

            # multi-task classification and segmentation loss
            self.bce_loss = torch.nn.BCEWithLogitsLoss()
            self.seg_bce_loss = torch.nn.BCEWithLogitsLoss()
            self.seg_dice_loss = smp.losses.DiceLoss(mode='binary')

        elif 'texturenet' in self.hparams.model.lower():
            self.hparams.in_channels = 32 # PCA texture features
            self.net = TextureNet(
                in_channels=self.hparams.in_channels, 
                num_labels=1, 
                fc_dim=128
            )

            # classification loss
            self.bce_loss = torch.nn.BCEWithLogitsLoss()

    def classification_loss(self, logit, label):
        # standard binary cross entropy loss
        return self.bce_loss(logit, label)
    
    def segmentation_loss(self, mask_logit, mask_label, alpha=0.5):
        # segmentation loss as implemented in Tomasz Włodarczy et. al. [1]
        # [1] Spontaneous preterm birth prediction using convolutional neural networks
        return alpha * self.seg_bce_loss(mask_logit, mask_label) + \
            (1-alpha) * self.seg_dice_loss(mask_logit, mask_label)

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(), 
            lr=self.hparams.lr, momentum=0.9,
            weight_decay=self.hparams.weight_decay,
            )

        callback_mode = "min" if self.hparams.monitor.lower() == "loss" else "max"

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode=callback_mode, 
            factor=self.hparams.lr_decline_factor,
            patience=self.hparams.lr_decline_patience, 
            verbose=True
        )

        return {"optimizer": opt,
                "lr_scheduler":{
                    "scheduler": sched,
                    "interval": "epoch",
                    "monitor": f"val/{self.hparams.monitor}",
                    "reduce_on_plateau":True,
                }
        }
    
    def forward(self, inputs):
        outputs = self.net(inputs)
        if 'mt-unet' in self.hparams.model.lower():
            return outputs['logit'], outputs['seg_logit']
        else:
            return outputs['logit']
        
    def step(self, inputs):
        outputs = self.net(inputs)
        logit = outputs['logit'].squeeze(-1)
        label = inputs['label'].float()

        logs = {}
        if 'sa-sononet' in self.hparams.model.lower() or 'texturenet' in self.hparams.model.lower():
            loss = self.classification_loss(logit, label)
            logs["loss"] = loss
        else:
            mask_label = inputs['binary_mask']
            mask_logit = outputs['seg_logit']

            loss_cls = self.classification_loss(logit, label)
            loss_seg = self.segmentation_loss(mask_logit, mask_label)

            tp, fp, fn, tn = smp.metrics.get_stats(
                mask_logit, mask_label.round().long(), 
                mode='binary',
                threshold=0.5
            )

            iou_score = smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro')
            loss = loss_cls + loss_seg
            loss = loss.float()
            logs["loss"] = loss
            logs["loss_seg"] = loss_seg
            logs["loss_cls"] = loss_cls
            logs["IoU"] = iou_score
        
        auc = auroc(logit, label.long(), task='binary')
        spe = specificity(logit, label.long(), task='binary', threshold=0.5)
        rec = recall(logit, label.long(), task='binary', threshold=0.5)
        acc = accuracy(logit, label.long(), task='binary', threshold=0.5)

        logs["AUROC"] = auc
        logs["ACC"] = acc
        logs["SEN"] = rec
        logs["SPE"] = spe

        return loss, logit, logs
    
    def training_step(self, batch, batch_idx):
        loss, outs, logs = self.step(batch)
        self.log_dict({f"train/{k}": v for k,v in logs.items()},
                       batch_size=batch['label'].size(0),
                       on_step=False, on_epoch=True)
                
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, outs, logs = self.step(batch)
        self.log_dict({f"val/{k}": v for k,v in logs.items()}, 
                      batch_size= batch['label'].size(0),
                      on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, out, logs = self.step(batch)
        self.log_dict({f"test/{k}": v for k,v in logs.items()},
                       batch_size= batch['label'].size(0),
                       on_step=False, on_epoch=True)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):

        ### Model spesific arguments ###
        parser = parent_parser.add_argument_group("Spontaneous Preterm Birth Classification Model")

        parser.add_argument(
            "--model", type=str, default="SA-SonoNet-32", help="model name: Default: SA-SonoNet-32"
        )

        parser.add_argument(
            "--in_channels", type=int, default=1, help="number of input channels. Default: 8 for SA-SonoNet"
        )

        parser.add_argument(
            "--lr", type=float, default=1e-3, help="learning rate (lr). Default: 1e-3"
        )

        parser.add_argument(
            "--lr_decline_patience", type=int, default=10, help="decay lr after x epochs of no improvement. Default: 10"
        )

        parser.add_argument(
            "--lr_decline_factor", type=float, default=0.75, help="lr decline factor. Default: 0.75"
        )

        parser.add_argument(
            "--weight_decay", type=float, default=1e-4, help="L2-regularizaion: Default: 1e-4"
        )
        
        return parent_parser


 
