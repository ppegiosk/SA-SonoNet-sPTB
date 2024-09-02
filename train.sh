#!/bin/bash

# loop through each fold index
folds=("fold_1" "fold_1B" "fold_2" "fold_2B" "fold_3" "fold_3B" "fold_4" "fold_4B" "fold_5" "fold_5B")

# SA-SonoNet
for fold in "${folds[@]}"
do
    python -m src.pretermbirth_model_train --split_index $fold --max_epochs 200 --batch_size 64 --accelerator gpu --gpu_id 1 --default_root_dir models/sa-sononet --model SA-SonoNet-32 --in_channels 8
done

# TextureNet
for fold in "${folds[@]}"
do
    python -m src.pretermbirth_model_train --split_index $fold --max_epochs 200 --batch_size 64 --accelerator gpu --gpu_id 1 --default_root_dir models/mt-unet --model MT-UNet
done

# MT-UNet
for fold in "${folds[@]}"
do
    python -m src.pretermbirth_model_train --split_index $fold --max_epochs 200 --batch_size 64 --accelerator gpu --gpu_id 1 --default_root_dir models/texturenet --model texturenet--dataset texture
done

