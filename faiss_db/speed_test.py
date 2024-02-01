from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import faiss
import sys
import argparse
import string
import math
import os

def init_model(args):
    return SentenceTransformer(args.model_name, device=args.device)

def get_dim(model):
    return model.state_dict()["0.auto_model.embeddings.word_embeddings.weight"].size()[1]


def faiss_index(d, args):
    index = faiss.IndexFlatIP(d)
    index_wrapper = faiss.IndexIDMap(index)
    if args.device != torch.device("cpu"):
        res = faiss.StandardGpuResources()
        final_index = faiss.index_cpu_to_gpu(res, 0, index_wrapper)
    return final_index

def normalize(embs):
    norm = np.linalg.norm(embs, axis=1)
    norm = norm.reshape(norm.shape[0],1)
    norm[norm==0.0] = 1.0
    return embs / norm

def index(model, args):
    eof = False
    pre_embs = []
    post_embs = []
    prev = args.last_pos
    with open(args.data_path, "r") as f:
        f.seek(prev)
        for i in range(args.batch_size):
            sent = f.readline()
            offset = len(sent)
            if sent != "":
                sent = sent.strip()
            else:
                eof = True
                break
            cur_pre_embs, cur_post_embs = encode(sent, model, args)
            pre_embs.extend(cur_pre_embs)
            post_embs.extend(cur_post_embs)
            
            prev += offset
    args.last_pos = prev
    args.eof = eof
    if pre_embs:
        return np.stack(normalize(pre_embs), axis=0), np.stack(normalize(post_embs), axis=0)
    else:
        return None, None

def encode(sent, model, args):
    pre_embs = []
    post_embs = []
    ngram = args.ngram
    word_list = sent.translate(str.maketrans("", "", string.punctuation)).split()
    seq_len = len(word_list)
    iters = math.ceil(seq_len / ngram)
    for i in range(iters - 1):
        pre = " ".join(word_list[:(i + 1) * ngram])
        post = " ".join(word_list[(i + 1) * ngram:])
        pre_emb = model.encode(pre)
        post_emb = model.encode(post)
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
    return pre_embs, post_embs

def main():
    parser = argparse.ArgumentParser(description="encoder")
    parser.add_argument("--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", required=False)
    parser.add_argument("--data_path", type=str, default="data.txt", required=False)
    parser.add_argument("--batch_size", type=int, default=32, required=False)
    parser.add_argument("--ngram", type=int, default=2, required=False)
    parser.add_argument("--out_folder", type=str, default="indices", required=False)
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    args.last_pos = 0
    args.eof = False
    print("device available:", args.device)
    model = init_model(args)
    print("on", model.device)
    
    model.to(args.device)
    print("model using device:", model.device)
    emb_dim  = get_dim(model)
    print("embedding dimension:", emb_dim)
    pre_index = faiss_index(emb_dim, args)
    post_index = faiss_index(emb_dim, args)
    start_id = 0
    iter = 0
    while not args.eof:
        pre_embs, post_embs = index(model, args)
        if pre_embs is not None:
            end_id = start_id + pre_embs.shape[0]
            ids = np.arange(start_id, end_id)
            pre_index.add_with_ids(pre_embs, ids)
            post_index.add_with_ids(post_embs, ids)
            start_id = end_id
        iter += 1
        if not iter % 100:
            print("indexed vectors:", pre_index.ntotal)
    if args.device != torch.device("cpu"):
        pre_index = faiss.index_gpu_to_cpu(pre_index)
        post_index = faiss.index_gpu_to_cpu(post_index)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    faiss.write_index(pre_index, os.path.join(args.out_folder, "prefix_index.index"))
    faiss.write_index(post_index, os.path.join(args.out_folder, "postfix_index.index"))