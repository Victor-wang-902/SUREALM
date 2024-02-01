import torch
import numpy as np
import faiss
import sys
import argparse
import string
import math
import os
import pickle
from torchtext.data.utils import get_tokenizer
from transformers import AutoTokenizer, AutoModel
from utils import encode, normalize_np
import json
from shutil import copyfile
import transformers

def init_model(args):
    model = AutoModel.from_pretrained(args.model_name, is_decoder=False)
    return model


def new_faiss_index(d, args):
    index = faiss.IndexFlatIP(d)
    final_index = faiss.IndexIDMap2(index)
    if args.device != torch.device("cpu"):
        res = faiss.StandardGpuResources()
        final_index = faiss.index_cpu_to_gpu(res, 0, final_index)
    return final_index


def prepare_index(file, args):
    pre_embs = []
    post_embs = []
    emb_map = []
    str_map = []
    id = 1
    ngram = file["ngram"]
    with open(os.path.join(args.data_folder, file["name"], "train.txt"), "r") as f:
        sents = [sent for sent in f]
        file["lines"] = len(sents)
    for i, sent in enumerate(sents):
        sent = sent.strip()
        if not args.no_skip:
            cur_pre_embs, cur_post_embs, cur_strs = skip_encode(sent, args, ngram)
        else:
            cur_pre_embs, cur_post_embs, cur_strs = naive_encode(sent, args) #doesn't work anymore
        emb_map.append(list(range(id,len(cur_pre_embs) + id)))
        str_map.extend(cur_strs)
        pre_embs.extend(cur_pre_embs)
        post_embs.extend(cur_post_embs)
        id += len(cur_post_embs)
        if (i+1) % 100 == 0:
            print("sentence", i)
    
    val_pre_embs = []
    val_emb_map = []
    id = 1
    with open(os.path.join(args.data_folder, file["name"], "eval.txt"), "r") as f:
        sents = [sent for sent in f]
    for i, sent in enumerate(sents):
        sent = sent.strip()
        if not args.no_skip:
            cur_pre_embs, _, _ = skip_encode(sent, args, ngram)
        else:
            cur_pre_embs, _, _ = naive_encode(sent, args)
        val_emb_map.append(list(range(id, len(cur_pre_embs) + id)))
        val_pre_embs.extend(cur_pre_embs)
        id += len(cur_pre_embs)        

    if pre_embs:
        return emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs 
        #return np.stack(normalize(pre_embs), axis=0), np.stack(normalize(post_embs), axis=0), strs
    else:
        return None, None, None, None, None, None

def skip_encode(sent, args, ngram):
    pre_embs = []
    post_embs = []
    strs = []
    inputs = args.tokenizer(sent, truncation=False)
    sos = inputs["input_ids"][0]
    eos = inputs["input_ids"][-1]
    word_list = inputs["input_ids"][1:-1]
    seq_len = len(word_list)
    iters = math.ceil(seq_len / ngram) - 2
    for i in range(iters):
        pre = {"input_ids": torch.tensor([sos] + word_list[:(i + 1) * ngram], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1)}
        post = {"input_ids": torch.tensor(word_list[(i + 2) * ngram:] + [eos], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1)}
        pre_emb = encode(pre, args).reshape(-1)

        post_emb = encode(post, args).reshape(-1)
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
        #print(pre["input_ids"].reshape(-1))
        pre_str = args.tokenizer.decode(pre["input_ids"].reshape(-1))
        post_str = args.tokenizer.decode(post["input_ids"].reshape(-1))
        strs.append((pre_str, post_str))
    return pre_embs, post_embs, strs

def naive_encode(sent, args):
    pre_embs = []
    post_embs = []
    if args.str_map:
        strs = []
    ngram = args.ngram
    word_list = sent.translate(str.maketrans("", "", string.punctuation)).split()
    seq_len = len(word_list)
    iters = math.ceil(seq_len / ngram)
    for i in range(iters - 1):
        pre = " ".join(word_list[:(i + 1) * ngram])
        post = " ".join(word_list[(i + 1) * ngram:])
        pre_emb = args.model.encode(pre).reshape(-1)
        post_emb = args.model.encode(post).reshape(-1)
        print(pre_emb.shape)
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
        if args.str_map:
            strs.append((pre, post))
    if args.str_map:
        return pre_embs, post_embs, strs
    return pre_embs, post_embs

def convert_embs(args, pre_embs=None, post_embs=None):
    if pre_embs is not None:
        ids = np.arange(1, len(pre_embs) + 1)
        pre_embs = np.stack(pre_embs, axis=0)
        pre_pad = np.zeros((1,pre_embs.shape[1]))
        #pre_pad = np.mean(pre_embs, axis=0).reshape(1,-1)
        #print(pre_embs)
        pre_embs_index = normalize_np(pre_embs)
        #print(pre_embs_index)
        args.pre_index.add_with_ids(pre_embs_index, ids)
        pre_embs = np.concatenate([pre_pad, pre_embs])
        pre_embs = pre_embs.astype(np.float32)
        pre_embs = torch.from_numpy(pre_embs)
        pre_embs = torch.nn.Embedding.from_pretrained(pre_embs)
    if post_embs is not None:
        post_embs = np.stack(post_embs, axis=0)
        #post_embs = normalize(post_embs)
        post_pad = np.zeros((1,post_embs.shape[1]))
        #post_pad = np.mean(post_embs, axis=0).reshape(1,-1)
        post_embs = np.concatenate([post_pad, post_embs])
        post_embs = post_embs.astype(np.float32)
        post_embs = torch.from_numpy(post_embs)
        post_embs = torch.nn.Embedding.from_pretrained(post_embs)
    return pre_embs, post_embs

def main():
    parser = argparse.ArgumentParser(description="encoder")
    parser.add_argument("--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", required=False)
    parser.add_argument("--data_folder", type=str, required=True)
    #parser.add_argument("--batch_size", type=int, default=0, required=False)
    parser.add_argument("--out_folder", type=str, default="processed_data", required=False)
    parser.add_argument("--cpu", action="store_true")
    #parser.add_argument("--str_map", action="store_true")
    parser.add_argument("--no_skip", action="store_true", default=False)
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.cpu:
        args.device = torch.device("cpu")
    print("device available:", args.device)
    args.model = init_model(args)
    args.model.to(args.device)
    args.model.eval()
    print("model using device:", args.model.device)
    emb_dim  = args.model.config.hidden_size
    print("embedding dimension:", emb_dim)
    args.raw_data_config = os.path.join(args.data_folder, "data_config.json")
    with open(args.raw_data_config, "r") as f:
        data_config = json.load(f)
    args.files = [data for data in data_config]
    args.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for file in args.files:
        args.pre_index = new_faiss_index(emb_dim, args)
        #start_id = 0
        #iteration = 0
    
        emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs = prepare_index(file, args)
        #print(pre_embs)
        pre_embs, post_embs = convert_embs(args, pre_embs, post_embs)
        _, val_pre_embs = convert_embs(args, post_embs=val_pre_embs)
        '''        
        while not args.eof:
            if args.str_map:
                pre_embs, post_embs, strs = prepare_index(model, args)
            else:
                pre_embs, post_embs = prepare_index(model, args)
            if pre_embs is not None:
                end_id = start_id + pre_embs.shape[0]
                ids = np.arange(start_id, end_id)
                pre_index.add_with_ids(pre_embs, ids)
                if args.str_map:
                    suffix_map.extend(strs)
                suffix_embeddings.append(post_embs)
                start_id = end_id
            iteration += 1
            if not iteration % 100:
                print("indexed vectors:", pre_index.ntotal)
                '''
        #suffix_embeddings = np.concatenate(suffix_embeddings, axis=0)
        #if args.device != torch.device("cpu"):
        #    pre_index = faiss.index_gpu_to_cpu(pre_index)
        #    post_index = faiss.index_gpu_to_cpu(post_index)
        folder_name = os.path.splitext(file["name"])[0] + "_" + str(file["ngram"]) + "gram"
        original_name = file["name"]
        file["name"] = folder_name
        os.makedirs(os.path.join(args.out_folder, folder_name))
        faiss.write_index(args.pre_index, os.path.join(args.out_folder, folder_name, "prefix_index.index"))  
        torch.save(pre_embs, os.path.join(args.out_folder, folder_name, "prefix_embeddings.pt"))
        torch.save(post_embs, os.path.join(args.out_folder, folder_name, "suffix_embeddings.pt"))
        torch.save(val_pre_embs, os.path.join(args.out_folder, folder_name, "val_prefix_embeddings.pt"))
        with open(os.path.join(args.out_folder, folder_name, "embedding_mappings.pkl"), "wb") as f:
            pickle.dump(emb_map, f)
        with open(os.path.join(args.out_folder, folder_name, "val_embedding_mappings.pkl"), "wb") as f:
            pickle.dump(val_emb_map, f)
        with open(os.path.join(args.out_folder, folder_name, "string_mappings.pkl"), "wb") as f:
            pickle.dump(str_map, f)
        copyfile(os.path.join(args.data_folder, original_name, "train.txt"), os.path.join(args.out_folder, folder_name, "data.txt"))
        copyfile(os.path.join(args.data_folder, original_name, "eval.txt"), os.path.join(args.out_folder, folder_name, "data_eval.txt"))

        print("total vectors indexed for", file["name"],":", args.pre_index.ntotal)
    if os.path.exists(os.path.join(args.out_folder, "data_config.json")):
        with open(os.path.join(args.out_folder, "data_config.json"), "r") as f:
            existing_json = json.load(f)
    else:
        existing_json = []
    existing_json.extend(args.files)
    args.files = existing_json
    with open(os.path.join(args.out_folder, "data_config.json"), "w") as f:
        json.dump(args.files, f)

if __name__ == "__main__":
    main()
