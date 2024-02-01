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
from utils import encode, normalize_np, encode_no_pool, mean_pooling_norm, weighted_pooling
import json
from shutil import copyfile
import transformers
import torch.multiprocessing as mp

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
    window_size = file["window_size"]
    with open(os.path.join(args.data_folder, file["name"], "train.txt"), "r") as f:
        sents = [sent.strip() for sent in f if sent.strip() != ""]
        file["lines"] = len(sents)
    if args.local_pooling:
        batch_size = args.pooling_bsz
        for i in range(0, len(sents), batch_size):
            #print("new batch")
            #print("start id", id)
            batched_sents = sents[i:i+batch_size]
            true_len = len(batched_sents)
            cur_pre_embs, cur_post_embs, cur_strs = pooling_encode(batched_sents, ngram, true_len, args)
            #print("after pooling:")
            num_embs = [len(x) for x in cur_pre_embs]
            num_embs = [0] + num_embs
            #print("num_embs", num_embs)
            cur_emb_map = torch.cumsum(torch.tensor(num_embs), dim=0).tolist()
            #print("cur_emb_map", cur_emb_map)
            emb_map.extend([list(range(id + cur_emb_map[i], id + cur_emb_map[i+1])) for i in range(len(cur_emb_map) - 1)])
            #print("emb_map", emb_map)
            cur_strs = [ele for l in cur_strs for ele in l]
            cur_pre_embs = [ele for l in cur_pre_embs for ele in l]
            cur_post_embs = [ele for l in cur_post_embs for ele in l]
            #print("cur_strs", cur_strs)
            #print("cur_pre_embs len", len(cur_pre_embs))
            #print("cur_post_embs len", len(cur_post_embs))
            str_map.extend(cur_strs)
            pre_embs.extend(cur_pre_embs)
            post_embs.extend(cur_post_embs)
            #print("str_map", str_map)
            #print("pre_embs len", len(cur_pre_embs))
            #print("post_embs len", len(cur_post_embs))
            id += cur_emb_map[-1]
            #print("end id", id)
            #print("batch done")
            if (i+1) % 100 == 0:
                print("sentence", i * true_len)
    else:
    
        for i, sent in enumerate(sents):
            #sent = " ".join(sent.strip().split()[:args.tokenizer.max_length])
            #if len(sent.split()) > args.tokenizer.max_length
            if not args.no_skip:
                cur_pre_embs, cur_post_embs, cur_strs = skip_encode_2(sent, ngram, window_size, args)
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
            cur_pre_embs, _, _ = skip_encode_2(sent, ngram, 0, args)
        else:
            cur_pre_embs, _, _ = naive_encode(sent, args)
        val_emb_map.append(list(range(id, len(cur_pre_embs) + id)))
        val_pre_embs.extend(cur_pre_embs)
        id += len(cur_pre_embs)        
    if os.path.exists(os.path.join(args.data_folder, file["name"], "test.txt")):
        test_pre_embs = []
        test_emb_map = []
        id = 1
        with open(os.path.join(args.data_folder, file["name"], "test.txt"), "r") as f:
            sents = [sent for sent in f]
        for i, sent in enumerate(sents):
            sent = sent.strip()
            if not args.no_skip:
                cur_pre_embs, _, _ = skip_encode_2(sent, ngram, 0, args)
            else:
                cur_pre_embs, _, _ = naive_encode(sent, args)
            test_emb_map.append(list(range(id, len(cur_pre_embs) + id)))
            test_pre_embs.extend(cur_pre_embs)
            id += len(cur_pre_embs)        
        if pre_embs:
            return emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs, test_emb_map, test_pre_embs
            #return np.stack(normalize(pre_embs), axis=0), np.stack(normalize(post_embs), axis=0), strs
        else:
            return None, None, None, None, None, None, None, None
    if pre_embs:
        return emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs, None, None
        #return np.stack(normalize(pre_embs), axis=0), np.stack(normalize(post_embs), axis=0), strs
    else:
        return None, None, None, None, None, None, None, None

def pooling_encode(sents, ngram, pooling_bsz=64, args=None, tokenizer=None, model=None):
    if args is not None:
        tokenizer = args.tokenizer
        model = args.model
    pre_embs = [[] for _ in range(pooling_bsz)]
    post_embs = [[] for _ in range(pooling_bsz)]
    strs = [[] for _ in range(pooling_bsz)]
    inputs = tokenizer(sents, truncation=True, padding="longest", return_tensors="pt")
    print("input lens", torch.sum(inputs["input_ids"]!=0, dim=1))
    seq_lens = torch.sum(inputs["attention_mask"], 1) - 2
    print("actual lens", seq_lens)
    iters = torch.ceil(seq_lens / ngram).int() - 2
    print("iters", iters)
    max_iter = torch.max(iters).item()
    if args is not None:
        sent_emb = encode_no_pool(inputs, args)
    else:
        sent_emb = encode_no_pool(inputs, model=model)
    print("sent_emb dim", sent_emb.shape)
    for i in range(max_iter):
        to_add = iters > i
        print("iteration", i)
        print("to_add", to_add)
        pre_emb = sent_emb[:,:(i + 1) * ngram + 1,:]
        post_emb = sent_emb[:,(i + 2) * ngram + 1:,:]
        print("pre_emb dim", pre_emb.shape)
        print("post_emb dim", post_emb.shape)

        pre_emb = mean_pooling_norm(pre_emb, inputs["attention_mask"][:,:(i + 1) * ngram + 1])
        post_emb = mean_pooling_norm(post_emb, inputs["attention_mask"][:,(i + 2) * ngram + 1:])
        print("after mean pooling")
        print("pre_emb dim", pre_emb.shape)
        print("post_emb dim", post_emb.shape)
        for j in range(pooling_bsz):
            if to_add[j]:
                print("add to", j, "th sent")
                pre_embs[j].append(pre_emb[j])
                post_embs[j].append(post_emb[j])
                strs[j].append((tokenizer.decode(inputs["input_ids"][j,:(i + 1) * ngram + 1]), tokenizer.decode(inputs["input_ids"][j,(i + 2) * ngram + 1:])))
                print(j,"th slot now has:")
                print(len(pre_embs[j]), "pre embs")
                print(len(post_embs[j]), "post embs")
                print(strs[j])
            else:
                print("not adding to", j, "th sent")
                continue
    return pre_embs, post_embs, strs


def skip_encode_2(sent, ngram, window_size, args=None, tokenizer=None, model=None, dropoff=None):
    if args is not None:
        tokenizer = args.tokenizer
        model = args.model
        dropoff = args.dropoff

    pre_embs = []
    post_embs = []
    strs = []
    inputs = tokenizer(sent, truncation=True)
    sos = inputs["input_ids"][0]
    eos = inputs["input_ids"][-1]
    word_list = inputs["input_ids"][1:-1]
    seq_len = len(word_list)
    iters = math.ceil(seq_len / ngram) - 2
    position_ids = torch.arange(0,len(inputs["input_ids"]),dtype=torch.long)
    for i in range(iters):
        pre = {"input_ids": torch.tensor([sos] + word_list[:(i + 1) * ngram], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1)}
        post = {"input_ids": torch.tensor(word_list[(i + 2) * ngram:] + [eos], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1)}
        if args.preserve_position_embedding:
            post["position_ids"] = position_ids[(i + 2) * ngram + 1:]
        if args is not None:
            pre_emb = encode(pre, args).reshape(-1)
            post_emb = encode_no_pool(post, args)
        else:
            pre_emb = encode(pre, model=model).reshape(-1)
            post_emb = encode_no_pool(post, model=model)
        if window_size != 0: # perform windowed dropoff
            post_emb = weighted_pooling(post_emb, post["attention_mask"], window_size=window_size, dropoff=dropoff).reshape(-1)
        else:
            post_emb = mean_pooling_norm(post_emb, post["attention_mask"]).reshape(-1)
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
        #print(pre["input_ids"].reshape(-1))
        pre_str = tokenizer.decode(pre["input_ids"].reshape(-1))
        post_str = tokenizer.decode(post["input_ids"].reshape(-1))
        strs.append((pre_str, post_str))
    return pre_embs, post_embs, strs

def skip_encode(sent, ngram, args=None, tokenizer=None, model=None):
    if args is not None:
        tokenizer = args.tokenizer
        model = args.model
    pre_embs = []
    post_embs = []
    strs = []
    inputs = tokenizer(sent, truncation=True)
    sos = inputs["input_ids"][0]
    eos = inputs["input_ids"][-1]
    word_list = inputs["input_ids"][1:-1]
    seq_len = len(word_list)
    iters = math.ceil(seq_len / ngram) - 2
    for i in range(iters):
        pre = {"input_ids": torch.tensor([sos] + word_list[:(i + 1) * ngram], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][:(i + 1) * ngram + 1], dtype=torch.int64).view(1,-1)}
        post = {"input_ids": torch.tensor(word_list[(i + 2) * ngram:] + [eos], dtype=torch.int64).view(1,-1), "token_type_ids": torch.tensor(inputs["token_type_ids"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1), "attention_mask": torch.tensor(inputs["attention_mask"][(i + 2) * ngram + 1:], dtype=torch.int64).view(1,-1)}
        if args is not None:
            pre_emb = encode(pre, args).reshape(-1)
            post_emb = encode(post, args).reshape(-1)
        else:
            pre_emb = encode(pre, model=model).reshape(-1)
            post_emb = encode(post, model=model).reshape(-1)
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
        #print(pre["input_ids"].reshape(-1))
        pre_str = tokenizer.decode(pre["input_ids"].reshape(-1))
        post_str = tokenizer.decode(post["input_ids"].reshape(-1))
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
        pre_embs_index = pre_embs
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

def prepare_fast_data(args, file, emb_map, pre_embs, post_embs, val_emb_map, val_pre_embs, test_emb_map, test_pre_embs):

    def search(index, pre_embeds, topk, emb_len=0):
        _, topk_indices = index.search(pre_embeds, emb_len + topk)
        topk_indices = np.array([np.setdiff1d(topk_indices[i,:], embs)[:topk] for i in range(topk_indices.shape[0])])
        topk_indices = topk_indices.reshape(-1)
        queue.put(topk_indices)

    final_pre_embs = []
    final_post_embs = []
    final_indices = []
    ngram = file["ngram"]
    topk = file["topk"]
    pre_embs.to(args.device)
    post_embs.to(args.device)
    if args.multi_index:
        with open(args.index_config, "r") as f:
            index_config = json.load(f)
        indices = []
        queue = mp.Queue()
        for i, index_path in enumerate(index_config):
            index = faiss.read_index(os.path.join(os.getcwd(), index_path["path"]))
            if not args.cpu:
                index = faiss.index_cpu_to_gpus_list(index, gpus=[i])
            indices.append(index)
        processes = []

    for embs in emb_map:
        if len(embs) == 0:
            final_pre_embs.append(pre_embs(torch.tensor([0]).to(args.device)).cpu())
            final_post_embs.append(post_embs(torch.tensor([0]).to(args.device)).cpu())
            final_indices.append(torch.tensor([0]))
            continue
        pre_embeds = pre_embs(torch.tensor(embs).to(args.device)).cpu().numpy()
        if args.multi_index:
            topk_indices_list = []
            for index in indices:
                p = mp.Process(target=search, args=(index, pre_embeds, topk//len(indices), len(embs)))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                topk_indices_list.append(queue.get())
            topk_indices = np.concat(topk_indices_list)
        else:
            _, topk_indices = args.pre_index.search(pre_embeds, len(embs) + topk)
            topk_indices = np.array([np.setdiff1d(topk_indices[i,:], embs)[:topk] for i in range(topk_indices.shape[0])])
            topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        final_indices.append(topk_indices)
        final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        torch.cuda.empty_cache()
    val_final_pre_embs = []
    val_final_post_embs = []
    val_final_indices = []
    for val_embs in val_emb_map:
        if len(val_embs) == 0:
            val_final_pre_embs.append(pre_embs(torch.tensor([0]).to(args.device)).cpu())
            val_final_post_embs.append(post_embs(torch.tensor([0]).to(args.device)).cpu())
            val_final_indices.append(torch.tensor([0]))
            continue
        pre_embeds = val_pre_embs(torch.tensor(val_embs).to(args.device)).cpu().numpy()
        if args.multi_index:
            topk_indices_list = []
            for index in indices:
                p = mp.Process(target=search, args=(index, pre_embeds, topk//len(indices)))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                topk_indices_list.append(queue.get())
            topk_indices = np.concat(topk_indices_list)
        else:
            _, topk_indices = args.pre_index.search(pre_embeds, topk)
            #topk_indices = np.array([np.setdiff1d(topk_indices[i:], val_embs) for i in range(topk_indices.shape[0])])[:,:topk]
            topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        val_final_indices.append(topk_indices)
        val_final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        val_final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        torch.cuda.empty_cache()
    #final_pre_embs = torch.concat(final_pre_embs, dim=0)
    #final_post_embs = torch.concat(final_post_embs, dim=0)
    #val_final_pre_embs = torch.concat(val_final_pre_embs, dim=0)
    #val_final_post_embs = torch.concat(val_final_post_embs, dim=0)
    #final_pre_embs = torch.nn.Embedding.from_pretrained(final_pre_embs)
    #final_post_embs = torch.nn.Embedding.from_pretrained(final_post_embs)
    #val_final_pre_embs = torch.nn.Embedding.from_pretrained(val_final_pre_embs)
    #val_final_post_embs = torch.nn.Embedding.from_pretrained(val_final_post_embs)
    
    if test_emb_map is None:
        return final_indices, final_pre_embs, final_post_embs, val_final_indices, val_final_pre_embs, val_final_post_embs, None, None, None

    test_final_pre_embs = []
    test_final_post_embs = []
    test_final_indices = []
    for test_embs in test_emb_map:
        if len(test_embs) == 0:
            test_final_pre_embs.append(pre_embs(torch.tensor([0]).to(args.device)).cpu())
            test_final_post_embs.append(post_embs(torch.tensor([0]).to(args.device)).cpu())
            test_final_indices.append(torch.tensor([0]))
            continue
        pre_embeds = test_pre_embs(torch.tensor(test_embs).to(args.device)).cpu().numpy()
        if args.multi_index:
            topk_indices_list = []
            for index in indices:
                p = mp.Process(target=search, args=(index, pre_embeds, topk//len(indices)))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
                topk_indices_list.append(queue.get())
            topk_indices = np.concat(topk_indices_list)
        else:
            _, topk_indices = args.pre_index.search(pre_embeds, topk)
            #topk_indices = np.array([np.setdiff1d(topk_indices[i:], val_embs) for i in range(topk_indices.shape[0])])[:,:topk]
            topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        test_final_indices.append(topk_indices)
        test_final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        test_final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        torch.cuda.empty_cache()
    return final_indices, final_pre_embs, final_post_embs, val_final_indices, val_final_pre_embs, val_final_post_embs, test_final_indices, test_final_pre_embs, test_final_post_embs
        

def main():
    parser = argparse.ArgumentParser(description="encoder")
    parser.add_argument("--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", required=False)
    parser.add_argument("--data_folder", type=str, required=True)
    parser.add_argument("--data_config", type=str, default="data_config.json", required=False)
    parser.add_argument("--local_pooling", action="store_true", default=False)
    parser.add_argument("--pooling_bsz", type=int, default=64, required=False)
    parser.add_argument("--multi_index", action="store_true", default=False)
    parser.add_argument("--index_config", type=str, required=False)
    parser.add_argument("--window_size", type=int, default=0, required=False)
    parser.add_argument("--dropoff", type=float, default=0., required=False)
    parser.add_argument("--preserve_position_embedding", action="store_true", default=True)
    parser.add_argument("--out_folder", type=str, default="processed_data", required=False)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--for_fast", action="store_true", default=False)
    parser.add_argument("--resume_fast", action="store_true", default=False)
    #parser.add_argument("--str_map", action="store_true")
    parser.add_argument("--no_skip", action="store_true", default=False)
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if args.multi_index:
        assert args.resume_fast, "only supports fast processing upon partially processed data with separate indices"
        mp.set_start_method("spawn", force=True)
    if args.cpu:
        args.device = torch.device("cpu")
    print("device available:", args.device)
    if args.resume_fast:
        args.processed_data_config = os.path.join(args.data_folder, args.data_config)
        with open(args.processed_data_config, "r") as f:
            data_config = json.load(f)
        args.files = [data for data in data_config]
    else:
        args.model = init_model(args)
        args.model.to(args.device)
        args.model.eval()
        print("model using device:", args.model.device)
        emb_dim  = args.model.config.hidden_size
        print("embedding dimension:", emb_dim)
        args.raw_data_config = os.path.join(args.data_folder, args.data_config)
        with open(args.raw_data_config, "r") as f:
            data_config = json.load(f)
        args.files = [data for data in data_config]
        args.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for file in args.files:
        if args.for_fast or args.resume_fast:
            folder_name = os.path.splitext(file["name"])[0] + "_" + str(file["ngram"]) + "gram" + "_" + "top" + str(file["topk"]) + "_" + "win" + str(file["window_size"])
        else:
            folder_name = os.path.splitext(file["name"])[0] + "_" + str(file["ngram"]) + "gram"
        if os.path.exists(os.path.join(args.out_folder, folder_name)):
            '''if folder_name == "dstc_1gram_top3":
                args.pre_index = faiss.read_index(os.path.join(args.out_folder, folder_name, "prefix_index.index"))  
                pre_embs = torch.load(os.path.join(args.out_folder, folder_name, "prefix_embeddings.pt"))
                post_embs = torch.load(os.path.join(args.out_folder, folder_name, "suffix_embeddings.pt"))
                val_pre_embs = torch.load(os.path.join(args.out_folder, folder_name, "val_prefix_embeddings.pt"))
                with open(os.path.join(args.out_folder, folder_name, "embedding_mappings.pkl"), "rb") as f:
                    emb_map = pickle.load(f)
                with open(os.path.join(args.out_folder, folder_name, "val_embedding_mappings.pkl"), "rb") as f:
                    val_emb_map = pickle.load(f)
                
                fast_idx, fast_pre_embs, fast_post_embs, val_fast_idx, val_fast_pre_embs, val_fast_post_embs = prepare_fast_data(args, file, emb_map, pre_embs, post_embs, val_emb_map, val_pre_embs)
                with open(os.path.join(args.out_folder, folder_name, "fast_indices.pkl"), "wb") as f:
                    pickle.dump(fast_idx, f)
                with open(os.path.join(args.out_folder, folder_name, "val_fast_indices.pkl"), "wb") as f:
                    pickle.dump(val_fast_idx, f)
                with open(os.path.join(args.out_folder, folder_name, "fast_prefix_embeddings.pkl"), "wb") as f:
                    pickle.dump(fast_pre_embs, f)
                with open(os.path.join(args.out_folder, folder_name, "fast_suffix_embeddings.pkl"), "wb") as f:
                    pickle.dump(fast_post_embs, f)
                with open(os.path.join(args.out_folder, folder_name, "val_fast_prefix_embeddings.pkl"), "wb") as f:
                    pickle.dump(val_fast_pre_embs, f)
                with open(os.path.join(args.out_folder, folder_name, "val_fast_suffix_embeddings.pkl"), "wb") as f:
                    pickle.dump(val_fast_post_embs, f)
                #torch.save(fast_pre_embs, os.path.join(args.out_folder, folder_name, "fast_prefix_embeddings.pt"))
                #torch.save(fast_post_embs, os.path.join(args.out_folder, folder_name, "fast_suffix_embeddings.pt"))
                #torch.save(val_fast_pre_embs, os.path.join(args.out_folder, folder_name, "val_fast_prefix_embeddings.pt"))
                #torch.save(val_final_post_embs, os.path.join(args.out_folder, folder_name, "val_fast_suffix_embeddings.pt"))'''
            print("skipping", folder_name)
            continue
        if args.resume_fast:
            if not args.multi_index:
                args.pre_index = faiss.read_index(os.path.join(args.data_folder, file["name"], "prefix_index.index"))
                if args.device != torch.device("cpu"):
                    res = faiss.StandardGpuResources()
                    args.pre_index = faiss.index_cpu_to_gpu(res, 0, args.pre_index)
            else:
                args.pre_index = None
            pre_embs = torch.load(os.path.join(args.data_folder, file["name"], "prefix_embeddings.pt"))
            post_embs = torch.load(os.path.join(args.data_folder, file["name"], "suffix_embeddings.pt"))
            val_pre_embs = torch.load(os.path.join(args.data_folder, file["name"], "val_prefix_embeddings.pt"))
            if os.path.exists(os.path.join(args.data_folder, file["name"], "test_prefix_embeddings.pt")):
                test_pre_embs = torch.load(os.path.join(args.data_folder, file["name"], "test_prefix_embeddings.pt"))
                with open(os.path.join(args.data_folder, file["name"], "test_embedding_mappings.pkl"), "rb") as f:
                    test_emb_map = pickle.load(f)
            else:
                test_pre_embs = None
                test_emb_map = None
            with open(os.path.join(args.data_folder, file["name"], "embedding_mappings.pkl"), "rb") as f:
                emb_map = pickle.load(f)
            with open(os.path.join(args.data_folder, file["name"], "val_embedding_mappings.pkl"), "rb") as f:
                val_emb_map = pickle.load(f)
            os.makedirs(os.path.join(args.out_folder, folder_name))
            original_name = file["name"]
            file["name"] = folder_name
            copyfile(os.path.join(args.data_folder, original_name, "data.txt"), os.path.join(args.out_folder, folder_name, "data.txt"))
            copyfile(os.path.join(args.data_folder, original_name, "data_eval.txt"), os.path.join(args.out_folder, folder_name, "data_eval.txt"))
            copyfile(os.path.join(args.data_folder, original_name, "embedding_mappings.pkl"), os.path.join(args.out_folder, folder_name, "embedding_mappings.pkl"))
            copyfile(os.path.join(args.data_folder, original_name, "prefix_embeddings.pt"), os.path.join(args.out_folder, folder_name, "prefix_embeddings.pt"))
            copyfile(os.path.join(args.data_folder, original_name, "prefix_index.index"), os.path.join(args.out_folder, folder_name, "prefix_index.index"))
            copyfile(os.path.join(args.data_folder, original_name, "suffix_embeddings.pt"), os.path.join(args.out_folder, folder_name, "suffix_embeddings.pt"))
            copyfile(os.path.join(args.data_folder, original_name, "val_embedding_mappings.pkl"), os.path.join(args.out_folder, folder_name, "val_embedding_mappings.pkl"))
            copyfile(os.path.join(args.data_folder, original_name, "val_prefix_embeddings.pt"), os.path.join(args.out_folder, folder_name, "val_prefix_embeddings.pt"))
            if os.path.exists(os.path.join(args.data_folder, original_name, "test_prefix_embeddings.pt")):
                copyfile(os.path.join(args.data_folder, original_name, "test_prefix_embeddings.pt"), os.path.join(args.out_folder, folder_name, "test_prefix_embeddings.pt"))
                copyfile(os.path.join(args.data_folder, original_name, "test_embedding_mappings.pkl"), os.path.join(args.out_folder, folder_name, "test_embedding_mappings.pkl"))
        else:
            args.pre_index = new_faiss_index(emb_dim, args)
            #start_id = 0
            #iteration = 0
        
            emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs, test_emb_map, test_pre_embs = prepare_index(file, args)
            #print(pre_embs)
            pre_embs, post_embs = convert_embs(args, pre_embs, post_embs)
            _, val_pre_embs = convert_embs(args, post_embs=val_pre_embs)
            if test_emb_map is not None:
                _, test_pre_embs = convert_embs(args, post_embs=test_pre_embs)
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
            original_name = file["name"]
            file["name"] = folder_name
            os.makedirs(os.path.join(args.out_folder, folder_name))
            faiss.write_index(args.pre_index, os.path.join(args.out_folder, folder_name, "prefix_index.index"))  
            torch.save(pre_embs, os.path.join(args.out_folder, folder_name, "prefix_embeddings.pt"))
            torch.save(post_embs, os.path.join(args.out_folder, folder_name, "suffix_embeddings.pt"))
            torch.save(val_pre_embs, os.path.join(args.out_folder, folder_name, "val_prefix_embeddings.pt"))
            if test_emb_map is not None:
                torch.save(test_pre_embs, os.path.join(args.out_folder, folder_name, "test_prefix_embeddings.pt"))
                with open(os.path.join(args.out_folder, folder_name, "test_embedding_mappings.pkl"), "wb") as f:
                    pickle.dump(test_emb_map, f)
            with open(os.path.join(args.out_folder, folder_name, "embedding_mappings.pkl"), "wb") as f:
                pickle.dump(emb_map, f)
            with open(os.path.join(args.out_folder, folder_name, "val_embedding_mappings.pkl"), "wb") as f:
                pickle.dump(val_emb_map, f)
            with open(os.path.join(args.out_folder, folder_name, "string_mappings.pkl"), "wb") as f:
                pickle.dump(str_map, f)
            copyfile(os.path.join(args.data_folder, original_name, "train.txt"), os.path.join(args.out_folder, folder_name, "data.txt"))
            copyfile(os.path.join(args.data_folder, original_name, "eval.txt"), os.path.join(args.out_folder, folder_name, "data_eval.txt"))

            print("total vectors indexed for", file["name"],":", args.pre_index.ntotal)

        if args.for_fast or args.resume_fast:
            fast_idx, fast_pre_embs, fast_post_embs, val_fast_idx, val_fast_pre_embs, val_fast_post_embs, test_fast_idx, test_fast_pre_embs, test_fast_post_embs = prepare_fast_data(args, file, emb_map, pre_embs, post_embs, val_emb_map, val_pre_embs, test_emb_map, test_pre_embs)
            if test_fast_idx is not None:
                with open(os.path.join(args.out_folder, folder_name, "test_fast_indices.pkl"), "wb") as f:
                    pickle.dump(test_fast_idx, f)
                with open(os.path.join(args.out_folder, folder_name, "test_fast_prefix_embeddings.pkl"), "wb") as f:
                    pickle.dump(test_fast_pre_embs, f)
                with open(os.path.join(args.out_folder, folder_name, "test_fast_suffix_embeddings.pkl"), "wb") as f:
                    pickle.dump(test_fast_post_embs, f)
            with open(os.path.join(args.out_folder, folder_name, "fast_indices.pkl"), "wb") as f:
                pickle.dump(fast_idx, f)
            with open(os.path.join(args.out_folder, folder_name, "val_fast_indices.pkl"), "wb") as f:
                pickle.dump(val_fast_idx, f)
            with open(os.path.join(args.out_folder, folder_name, "fast_prefix_embeddings.pkl"), "wb") as f:
                pickle.dump(fast_pre_embs, f)
            with open(os.path.join(args.out_folder, folder_name, "fast_suffix_embeddings.pkl"), "wb") as f:
                pickle.dump(fast_post_embs, f)
            with open(os.path.join(args.out_folder, folder_name, "val_fast_prefix_embeddings.pkl"), "wb") as f:
                pickle.dump(val_fast_pre_embs, f)
            with open(os.path.join(args.out_folder, folder_name, "val_fast_suffix_embeddings.pkl"), "wb") as f:
                pickle.dump(val_fast_post_embs, f)
            #torch.save(fast_pre_embs, os.path.join(args.out_folder, folder_name, "fast_prefix_embeddings.pt"))
            #torch.save(fast_post_embs, os.path.join(args.out_folder, folder_name, "fast_suffix_embeddings.pt"))
            #torch.save(val_fast_pre_embs, os.path.join(args.out_folder, folder_name, "val_fast_prefix_embeddings.pt"))
            #torch.save(val_final_post_embs, os.path.join(args.out_folder, folder_name, "val_fast_suffix_embeddings.pt"))
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
