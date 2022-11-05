#!/bin/bash

model_name = "bert-base-uncased"
data = "fast_processed_data"

python scripts/train_script.py \
    --model_path $model_name \
    --data_config data_config.json \
    --data_folder $data \
    --output distilbert_bert \
    --batch_size 64 \
    --epochs 200 \
    --lr 1e-5 \
    --save_head  \
    --save_epochs 1 \