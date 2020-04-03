#!/bin/bash

for i in 0 1
do
    python train.py \
        --device-ids 0,1 \
        --batch-size 6 \
        --fold $i \
        --workers 12 \
        --lr 0.0001 \
        --n-epochs 2 \
        --jaccard-weight 0.3 \
        --model UNet16 \
        --train_crop_height 1024 \
        --train_crop_width 1280 \
        --val_crop_height 1024 \
        --val_crop_width 1280
done
