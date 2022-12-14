import os
import math
import torch
import faiss
import numpy as np
import transformers
from transformers import AutoModel
import logging
import argparse
from typing import Dict, List

from ..utils import (
    encode, 
    encode_no_pool, 
    mean_pooling_norm, 
    weighted_pooling
)


def init_model(args: argparse.Namespace):
    model = AutoModel.from_pretrained(args.model_name, is_decoder=False)
    return model


def new_faiss_index(d: int, args: argparse.Namespace):
    index = faiss.IndexFlatIP(d)
    final_index = faiss.IndexIDMap2(index)
    return final_index


def index_to_gpu(index: faiss.IndexIDMap2):
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    return index


def index_to_cpu(index: faiss.IndexIDMap2):
    index = faiss.index_gpu_to_cpu(index)
    return index


def prepare_index(
    file: Dict[str, str], 
    args: argparse.Namespace
    ):
    '''
    Preprocess data files (train, validation, test)
    indicated in the json file `file`. Returns a tuple containing:
    emb_map: List[Tuple], mapping the i-th sentence from train data to a tuple of ids of its embedding pairs.
    str_map: List[Tuple], mapping the i-th sentence from train data to a tuple of prefix-suffix strings for debugging.
    pre_embs: List[torch.Tensor], stores all the prefix embeddings from train data.
    post_embs: List[torch.Tensor], stores all the suffix embeddings from train data.
    val_emb_maps: List[Tuple], same as `emb_map` but for validation data.
    val_pre_embs: List[torch.Tensor], same as `pre_embs` but for validation data.
    test_emb_maps: List[Tuple], same as `emb_map` but for test data.
    test_pre_embs: List[torch.Tensor], same as `pre_embs` but for test data.
    Parameters: 
        file: dict, a json style dictionary containing the following fields:
            name: str, name of the folder containing the text files.
            ngram: int, frequency of retrieval.
            window_size: int, truncation window size.
            dropoff: float, optional, the dropoff discount factor once out of the truncation window, default 0.
        args: Namespace, Namespace object returned by argparse.ArgumentParser.parse_args().
    '''             
    ngram = file["ngram"]
    window_size = file["window_size"]
    if "dropoff" in file.keys():
        dropoff = file["dropoff"]
    else:
        dropoff = 0.
    pre_embs = []
    post_embs = []
    emb_map = []
    str_map = []
    id = 1
    with open(os.path.join(args.data_folder, file["name"], "train.txt"), "r") as f:
        sents = [sent.strip() for sent in f if sent.strip() != ""]
        file["lines"] = len(sents)
    for i, sent in enumerate(sents):
        cur_pre_embs, cur_post_embs, cur_strs = skip_encode(sent, ngram, window_size, dropoff, args)
        emb_map.append(list(range(id,len(cur_pre_embs) + id)))
        str_map.extend(cur_strs)
        pre_embs.extend(cur_pre_embs)
        post_embs.extend(cur_post_embs)
        id += len(cur_post_embs)
        if (i+1) % 100 == 0:
            print("sentence encoded", i)
    test_pre_embs = []
    test_emb_map = []
    id = 1
    with open(os.path.join(args.data_folder, file["name"], "test.txt"), "r") as f:
        logging.info("prepare_index test data in knowledge test: " +os.path.join(args.data_folder, file["name"], "test.txt"))
        sents = [sent for sent in f]
    for i, sent in enumerate(sents):
        sent = sent.strip()
        cur_pre_embs, _, _ = skip_encode(sent, ngram, 0, 0., args)
        test_emb_map.append(list(range(id, len(cur_pre_embs) + id)))
        test_pre_embs.extend(cur_pre_embs)
        id += len(cur_pre_embs)
    val_pre_embs = []
    val_emb_map = []
    id = 1
    with open(os.path.join(args.data_folder, file["name"], "val.txt"), "r") as f:
        sents = [sent for sent in f]
    for i, sent in enumerate(sents):
        sent = sent.strip()
        cur_pre_embs, _, _ = skip_encode(sent, ngram, 0, 0., args)
        val_emb_map.append(list(range(id, len(cur_pre_embs) + id)))
        val_pre_embs.extend(cur_pre_embs)
        id += len(cur_pre_embs)        
    if val_pre_embs or pre_embs or test_pre_embs:
        return emb_map, str_map, pre_embs, post_embs, val_emb_map, val_pre_embs, test_emb_map, test_pre_embs
    else:
        return None, None, None, None, None, None, None, None


def skip_encode(
    sent: str, 
    ngram: int, 
    window_size: int, 
    dropoff: float, 
    args: argparse.Namespace=None, 
    tokenizer: transformers.PreTrainedTokenizerFast=None, 
    model: transformers.PreTrainedModel=None, 
    tgt_tokenizer: transformers.PreTrainedTokenizerFast=None
    ):
    '''
    Function that encodes a sentence based on defined config. 
    Currently tested on models with model_type: bert, roberta, mpnet.
    Returns a tuple containing:
        pre_embs: List[torch.Tensor], prefix embeddings from the sentence.
        post_embs: List[torch.Tensor], suffix embeddings from the sentence.
        strs: List[Tuple], tuples of prefix-suffix strings from the sentence.
    Parameters:
        sent: str, the sentence to encode.
        ngram: int, frequency of retrieval.
        window_size: int, size of truncation window.
        dropoff: float, dropoff discount factor.
        args: Namespace, optional Namespace object returned by argparse.ArgumentParser.parse_args().
        tokenizer: PreTrainedTokenizerFast, optional Huggingface tokenizer of the encoding model. Only valid if args is not given.
        model: PreTrainedModel, optional Huggingface pretrained model as the encoder. Only valid if args is not given.
        tgt_tokenizer: PreTrainedTokenizerFast, optional Huggingface tokenizer for which the embeddings are prepared. Only valid if args is not given.
    '''
    if args is not None:
        tokenizer = args.tokenizer
        model = args.model
        tgt_tokenizer = args.target_tokenizer
    pre_embs = []
    post_embs = []
    strs = []
    # quick hack for bert tokenizers which don't have bos and eos tokens.
    sos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id  
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    if tgt_tokenizer is not None:
        inputs = tgt_tokenizer(sent, truncation=True, add_special_tokens=False)
    else:
        inputs = tokenizer(sent, truncation=True, add_special_tokens=False)
    word_list = inputs["input_ids"]
    seq_len = len(word_list)
    # calculates the number of iterations
    iters = math.ceil(seq_len / ngram) - 2
    for i in range(iters):  # TODO: use batch encoding to increase efficiency.
        if tgt_tokenizer is not None:  # decode target substring and encode using encoder tokenizer.
            pre_inputs = tokenizer(
                tgt_tokenizer.decode(
                    word_list[:(i + 1) * ngram], 
                    skip_special_tokens=True
                    ),
                truncation=True, 
                add_special_tokens=False
                )
            pre = {
                "input_ids": torch.tensor([sos] + pre_inputs["input_ids"], dtype=torch.int64).view(1,-1).to(args.device), 
                "attention_mask": torch.tensor([1] + pre_inputs["attention_mask"], dtype=torch.int64).view(1,-1).to(args.device)
                }
            post_inputs = tokenizer(
                tgt_tokenizer.decode(
                    word_list[(i + 2) * ngram:], 
                    skip_special_tokens=True
                    ), 
                truncation=True, 
                add_special_tokens=False
                )
            post = {
                "input_ids": torch.tensor(post_inputs["input_ids"] + [eos], dtype=torch.int64).view(1,-1).to(args.device), 
                "attention_mask": torch.tensor(post_inputs["attention_mask"] + [1], dtype=torch.int64).view(1,-1).to(args.device)
                }
            if not args.not_preserve_position_embedding:
                post["position_ids"] = torch.arange(
                    pre["input_ids"].shape[1] + 1, 
                    post["input_ids"].shape[1] + pre["input_ids"].shape[1] + 1, 
                    dtype=torch.long
                    ).to(args.device)
        else:  # direct encoding without additional decoding step
            position_ids = torch.arange(0,len(inputs["input_ids"]) + 2,dtype=torch.long)
            pre = {
                "input_ids": torch.tensor([sos] + word_list[:(i + 1) * ngram], dtype=torch.int64).view(1,-1).to(args.device), 
                "attention_mask": torch.tensor([1] + inputs["attention_mask"][:(i + 1) * ngram], dtype=torch.int64).view(1,-1).to(args.device)
                }
            post = {
                "input_ids": torch.tensor(word_list[(i + 2) * ngram:] + [eos], dtype=torch.int64).view(1,-1).to(args.device), 
                "attention_mask": torch.tensor(inputs["attention_mask"][(i + 2) * ngram:] + [1], dtype=torch.int64).view(1,-1).to(args.device)
                }
            if not args.not_preserve_position_embedding:
                post["position_ids"] = position_ids[(i + 2) * ngram + 1:].to(args.device)
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
        pre_emb = pre_emb.cpu()
        post_emb = post_emb.cpu()
        pre_embs.append(pre_emb)
        post_embs.append(post_emb)
        pre_str = tokenizer.decode(pre["input_ids"].reshape(-1))
        post_str = tokenizer.decode(post["input_ids"].reshape(-1))
        strs.append((pre_str, post_str))
    return pre_embs, post_embs, strs


def convert_embs(
    args: argparse.Namespace, 
    pre_embs: List[torch.Tensor]=None, 
    post_embs: List[torch.Tensor]=None
    ):
    '''
    Index prefix embeddings in FAISS index and convert stacked embedding tensors into torch.nn.Embedding.
    Returns a tuple containing:
        pre_embs: torch.nn.Embedding, optional prefix embeddings as torch.nn.Embedding if given.
        post_embs: torch.nn.Embedding, optional suffix embeddings as torch.nn.Embedding if given.
    Parameters:
        args: Namespace, returned by argparse.ArgumentParser.parse_args().
        pre_embs: list, optional prefix embedding tensors returned by prepare_index(...).
        post_embs: list, optional suffix embedding tensors returned by prepare_index(...).
    '''
    if pre_embs is not None:
        ids = np.arange(1, len(pre_embs) + 1)
        pre_embs = np.stack(pre_embs, axis=0)
        pre_pad = np.zeros((1,pre_embs.shape[1]))
        pre_embs_index = pre_embs
        args.pre_index.add_with_ids(pre_embs_index, ids)
        pre_embs = np.concatenate([pre_pad, pre_embs])
        pre_embs = pre_embs.astype(np.float32)
        pre_embs = torch.from_numpy(pre_embs)
        pre_embs = torch.nn.Embedding.from_pretrained(pre_embs)
    if post_embs is not None:
        post_embs = np.stack(post_embs, axis=0)
        post_pad = np.zeros((1,post_embs.shape[1]))
        post_embs = np.concatenate([post_pad, post_embs])
        post_embs = post_embs.astype(np.float32)
        post_embs = torch.from_numpy(post_embs)
        post_embs = torch.nn.Embedding.from_pretrained(post_embs)
    return pre_embs, post_embs


def prepare_fast_data(
    args: argparse.Namespace,
    file: Dict[str, str], 
    emb_map: List[tuple],
    pre_embs: torch.nn.Embedding, 
    post_embs: torch.nn.Embedding, 
    val_emb_map: List[tuple], 
    val_pre_embs: torch.nn.Embedding, 
    test_emb_map: List[tuple], 
    test_pre_embs: torch.nn.Embedding
    ):
    '''
    Precomputing all retrieval information to be used during training for (much) faster training. 
    Returns a tuple containing:
        final_indices: List[torch.Tensor], mapping the i-th sentence from train data to int tensors corresponding to its retrieved embedding ids.
        final_pre_embs: List[torch.Tensor], mapping the i-th sentence from train data to its retrieved prefix embedding tensors directly.
        final_post_embs: List[torch.Tensor], mapping the i-th sentence from train data to its retrieved suffix embedding tensors directly.
        val_final_indices: List[torch.Tensor], same as final_indices but for validation data.
        val_final_pre_embs: List[torch.Tensor], same as final_pre_embs but for validation data.
        val_final_post_embs: List[torch.Tensor], same as final_post_embs but for validation data.
        test_final_indices: List[torch.Tensor], same as final_indices but for test data.
        test_final_pre_embs: List[torch.Tensor], same as final_pre_embs but for test data.
        test_final_post_embs: List[torch.Tensor], same as final_post_embs but for test data.
    Parameters:
        args: Namespace, Namespace object returned from argparse.ArgumentParser.parse_args().
        file: dict, a json style dictionary containing the following fields:
            ngram: int, frequency of retrieval.
            topk: int, topk embeddings to retrieve for each step.
        emb_map: List[Tuple], mapping the i-th sentence from train data to a tuple of ids of its embedding pairs.
        pre_embs: List[torch.Tensor], stores all the prefix embeddings from train data.
        post_embs: List[torch.Tensor], stores all the suffix embeddings from train data.
        val_emb_map: List[Tuple], same as `emb_map` but for validation data.
        val_pre_embs: List[torch.Tensor], same as `pre_embs` but for validation data.
        test_emb_map: List[Tuple], same as `emb_map` but for test data.
        test_pre_embs: List[torch.Tensor], same as `pre_embs` but for test data.
    '''
    ngram = file["ngram"]
    topk = file["topk"]
    pre_embs.to(args.device)
    post_embs.to(args.device)
    val_pre_embs.to(args.device)
    args.pre_index = index_to_gpu(args.pre_index)
    if test_pre_embs is not None:
        test_pre_embs.to(args.device)
    final_pre_embs = []
    final_post_embs = []
    final_indices = []
    logging.info("start precomputation for train")
    count = 0
     # TODO: use a larger batch size for searching for better efficiency. 
     # Current implementation batch size is equal to the number of embeddings in each sentence.
    for embs in emb_map:
        if len(embs) == 0:  # in case a sentence is too short for any retrieval.
            final_pre_embs.append(pre_embs(torch.tensor([0]).to(args.device)).cpu())
            final_post_embs.append(post_embs(torch.tensor([0]).to(args.device)).cpu())
            final_indices.append(torch.tensor([0]))
            continue
        pre_embeds = pre_embs(torch.tensor(embs).to(args.device)).cpu().numpy()
        _, topk_indices = args.pre_index.search(pre_embeds, len(embs) + topk)
        # exclude the embedding retrieved from current sentence for train data
        topk_indices = np.array([np.setdiff1d(topk_indices[i,:], embs, assume_unique=True)[:topk] for i in range(topk_indices.shape[0])])
        topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        final_indices.append(topk_indices)
        final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        count += 1
        if count % 100 == 0:
            logging.info("sentence precomputed: "+ str(count))
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

        _, topk_indices = args.pre_index.search(pre_embeds, topk)
        topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        val_final_indices.append(topk_indices)
        val_final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        val_final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        torch.cuda.empty_cache()
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
        _, topk_indices = args.pre_index.search(pre_embeds, topk)
        topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        test_final_indices.append(topk_indices)
        test_final_pre_embs.append(pre_embs(topk_indices.to(args.device)).cpu())
        test_final_post_embs.append(post_embs(topk_indices.to(args.device)).cpu())
        torch.cuda.empty_cache()
    return final_indices, final_pre_embs, final_post_embs, val_final_indices, val_final_pre_embs, val_final_post_embs, test_final_indices, test_final_pre_embs, test_final_post_embs