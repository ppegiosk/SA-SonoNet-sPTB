import argparse
import glob
import os
import re
import torch
import yaml


def get_supported_datamodules():
    from src.datamodules.pretermbirth_image_datamodule import PretermBirthImageDatamdule
    from src.datamodules.pretermbirth_texture_datamodule import PretermBirthTextureDataModule
    # we can use spesific initilizations for a datamodule
    supported_datamodels = {"splits": (PretermBirthImageDatamdule,{'data':"splits"}),
                            "texture": (PretermBirthTextureDataModule,{'data':"splits"}),
                            "external": (PretermBirthImageDatamdule,{'data':"external"}),
    }

    return supported_datamodels


def load_hparams_from_logdir(logdir):
    
    with open(os.path.join(logdir, "hparams.yaml")) as file:
        hparams_dict = yaml.load(file, Loader=yaml.FullLoader)    
    hparams = argparse.Namespace(**hparams_dict)

    return hparams


def load_datamodule_from_name(dataset_name, **args):
    """Loads a Datamodule
    Args:
        dataset_name (str): Name of dataset
        batch_size (int, optional): Batch size. Defaults to 32.

    Returns:
        Datamodule
    """
    
    supported_datamodels = get_supported_datamodules()
    if dataset_name not in supported_datamodels:
        raise Exception(
            f"Dataset {dataset_name} unknown. Supported datasets: {supported_datamodels.keys()}"
        )

    # get data module and default args
    datamodule_cls, default_args = supported_datamodels[dataset_name]

    # merge args and default args
    for k, v in default_args.items():
        if k not in args.keys():
            args[k] = v

    datamodule = datamodule_cls(**args)

    return datamodule


def get_checkoint_path_from_logdir(model_logdir):
    epoch_to_checkpoint = {}
    regex = r".*epoch=([0-9]+)-step=[0-9]+.ckpt"
    checkpoint_files = glob.glob(
        os.path.join(model_logdir, "checkpoints", "*"))
    if len(checkpoint_files) == 0:
        raise Exception(
            f'Could not find any model checkpoints in {model_logdir}.')
    for fname in checkpoint_files:
        if re.match(regex, fname):
            epoch = re.search(regex, fname).group(1)
            epoch_to_checkpoint[int(epoch)] = fname
    return sorted(epoch_to_checkpoint.items(), key=lambda t: t[0])[-1][1]


def to_device(obj, device):
    """Maps a torch tensor, or a collection containing torch tesnors recursively onto the gpu

    Args:
        obj ([type]): [description]
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif hasattr(obj, "__iter__"):
        return [to_device(o, device) for o in obj]
    else:
        raise Exception(f"Do not know how to map object {obj} to {device}")