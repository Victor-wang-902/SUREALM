#!/bin/bash

model_checkpoint="model2/1"
model_type="gpt2"
knowledge="fast_processed_data/dstc_1gram_top8_win10"
sbert="sentence-transformers/multi-qa-distilbert-cos-v1"

python inference.py \
    --model_path $model_checkpoint \
    --model_type $model_type \
    --tokenizer_difference \
    --knowledge_path $knowledge \
    --max_length 20 \
    --max_context_length_per_k 64 \
    --topk 8 \
    --stride 1 \
    --interactive \
    --sbert_path $sbert