#!/bin/bash

model_name="bert-base-uncased"
data="fast_processed_data"
out_dir="model"

python train_script.py \
    --model_path $model_name \
    --data_config data_config.json \
    --data_folder $data \
    --output $out_dir \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-5 \
    --save_head  \
    --save_epochs 1 \