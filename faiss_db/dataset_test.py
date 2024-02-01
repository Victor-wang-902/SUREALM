import argparse
from transformers import AutoTokenizer
import os
import json
import tqdm
from dataset import load_dataset
from utils import AttentionMaskGenerator
import torch

if __name__ == "__main__":
    torch.set_printoptions(threshold=10_000)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", default="processed_data")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--external_embedding", action="store_true", default=False)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--max_context_length_per_k", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    args.tokenizer = AutoTokenizer.from_pretrained("modified_multi-qa-MiniLM-L6-cos-v1")
    with open(os.path.join(args.data_folder, "data_config.json"), "r") as fIn:
        data_config = json.load(fIn)
    args.files = []
    #dataset_indices = []
    args.total_lines = 0

    for data in data_config:
        args.files.append(data)
        #dataset_indices.extend([idx]*data['weight'])
        args.total_lines += data["lines"]
    mask_gen = AttentionMaskGenerator()
    for i, dataloader in enumerate(tqdm.tqdm(iter(load_dataset(args)))):
        for batch in dataloader:
            print(batch)
            break