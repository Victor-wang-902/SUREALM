#!/bin/bash

model_name="sentence-transformers/multi-qa-distilbert-cos-v1"
out_dir="fast_processed_data"
sbert="sentence-transformers/multi-qa-distilbert-cos-v1"

#set not_preserve_position_embedding if sentence transformer does not accept position embeddings
python encoder.py \
    --model_name $model_name \
    --data_folder data \
    --data_config data_config.json \
    --out_folder $out_dir \
    --not_preserve_position_embedding