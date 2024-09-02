import argparse
import numpy as np
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import random
import torch

from src.pretermbirth_model import PretermBirthModel
import src.util as util


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = PretermBirthModel.add_model_specific_args(parser)
    parser = add_program_level_args(parser)
    return parser


def add_program_level_args(parent_parser):
    parser = parent_parser.add_argument_group("Program Level Arguments")


    parser.add_argument(
        "--dataset", type=str, default="splits", help=f"Dataset. Options: {util.get_supported_datamodules().keys()}"
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="batchsize Default: 64"
    )

    parser.add_argument(
        "--split_index", type=str, default="fold_1", help="split index"
    )

    parser.add_argument(
        "--label", type=str, default="birth_before_week_37", help="birth_before_week_37"
    )
    
    parser.add_argument(
            "--img_size",
            nargs="+",
            type=int,
            default=[224, 288],
            help="image size. Defaut: [224, 288]",
        )

    parser.add_argument(
        "--early_stop_patience", type=int, default=40, help="Early stopping patience, in Epcohs. Default: 40",
    )

    parser.add_argument(
        "--monitor", type=str, default='loss', help="validation metric for monitoring",
    )

    # parser.add_argument(
    #     "--class_only", type=int, default=-1, help="not relevant for cls model"
    # )

    parser.add_argument(
        "--load_from_checkpoint", help="optional model checkpoint to initialize with"
    )

    parser.add_argument(
        "--load_weights", help="optional model checkpoint to initialize with"
    )

    parser.add_argument(
        "--notest", action="store_true", help="Set to not run test after training."
    )

    parser.add_argument(
        "--gpu_id", type=int, default=1, help="gpu id to run"
    )

    return parent_parser


def load_datamodule_from_params(hparams):
    datamodule = util.load_datamodule_from_name(
        dataset_name=hparams.dataset,
        batch_size=hparams.batch_size, 
        split_index=hparams.split_index,
        img_size= hparams.img_size,
        label=hparams.label,
        # class_only=hparams.class_only,
        )
    return datamodule, hparams


def load_model_from_hparams(hparams):
    if hparams.load_from_checkpoint:
        model = PretermBirthModel.load_from_checkpoint(
            checkpoint_path=hparams.load_from_checkpoint)
        hparams.resume_from_checkpoint = hparams.load_from_checkpoint
    else:
        model = PretermBirthModel(hparams)
    return model, hparams


def config_trainer_from_hparams(hparams):

    if hparams.monitor == "loss" : 
        callback_mode="min"
    else:
        callback_mode="max"
    
    print(f"callback: {hparams.monitor}-{callback_mode}")

    # save model with best validation loss
    checkpointing_callback = pl.callbacks.ModelCheckpoint(
        monitor=f"val/{hparams.monitor}", mode=callback_mode
    )
    # early stopping
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor=f"val/{hparams.monitor}", min_delta=0.00, 
        patience=hparams.early_stop_patience, verbose=True, 
        mode=callback_mode
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        hparams,
        callbacks=[checkpointing_callback, early_stop_callback, lr_monitor],
        #callbacks=[lr_monitor],
    )
    return trainer


def main(hparams):
    # set-up
    seed = 42
    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    os.environ["CUDA_VISIBLE_DEVICES"]= str(hparams.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")

    torch.set_float32_matmul_precision('medium')
    
    print(hparams)
    datamodule, hparams = load_datamodule_from_params(hparams)
    print('Train set: ', datamodule.train_dataloader().dataset.__len__())
    print('Validation set: ', datamodule.val_dataloader().dataset.__len__())
    print('Test set size: ', datamodule.test_dataloader().dataset.__len__())
    model, hparams = load_model_from_hparams(hparams)
    trainer = config_trainer_from_hparams(hparams)

    # train
    if hparams.load_weights:
        ckpt_path = hparams.load_weights
    else:
        ckpt_path = None
    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    # test
    if not hparams.notest:
        trainer.test(datamodule=datamodule, ckpt_path='best')


if __name__ == "__main__":
    parser = build_arg_parser()
    hparams = parser.parse_args()
    main(hparams)
    print("Training is finished!")
