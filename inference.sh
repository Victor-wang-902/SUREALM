#!/bin/bash

model_checkpoint = "model/18"
knowledge = "fast_processed_data/dstc_1gram_top8"
sbert = "sentence-transformers/multi-qa-distilbert-cos-v1"

python scripts/inference.py \
    --model_path $model_checkpoint \
    --knowledge_path $knowledge \
    --max_length 64 \
    --max_context_length_per_k 64 \
    --topk 8 \
    --stride 1 \
    --interactive \
    --sbert_path $sbert