import torch
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="embedding_saver")
    parser.add_argument("--input_path", type=str, required=True)
    args = parser.parse_args()
    dirname = os.path.dirname(args.input_path)
    with open(args.input_path, "rb") as f:
        embeddings = np.load(f)
    vocab_size = embeddings.shape[0]
    dim = embeddings.shape[1]
    emb_layer = torch.nn.embedding(vocab_size, dim)
    
