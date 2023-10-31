import os
import pickle
import argparse
from typing import Dict
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import transformers

from ..utils import AttentionMaskGenerator, NaiveAttentionMaskGenerator


class FastRetrievalDataset(Dataset):
    def __init__(
        self, 
        data: Dict[str, str], 
        args: argparse.Namespace, 
        val: bool=False, 
        ):
        '''
        Dataset for retrieval training. 
        Each example is a tuple containing the i-th sentence, its retrieved prefix and suffix embeddings.
        Parameters:
            data: dict, json style dictionary containing the following fields:
                name: str, name of the folder containing the text files.
                ngram: int, frequency of retrieval.
                topk: int, topk embeddings to retrieve for each step.
            val: bool, whether the dataset is for validation (or evaluation depending on whether args.eval_only=True).
        '''
        data_folder = args.data_folder
        self.val = val
        self.ngram = data["ngram"]
        self.topk = data["topk"]
        if self.val:
            if not args.eval_only:
                self.text_path = os.path.join(data_folder, data["name"], "data_val.txt")
                self.pre_path = os.path.join(data_folder, data["name"], "val_fast_prefix_embeddings.pkl")
                self.post_path = os.path.join(data_folder, data["name"], "val_fast_suffix_embeddings.pkl")
            else:
                self.text_path = os.path.join(data_folder, data["name"], "data_test.txt")
                self.pre_path = os.path.join(data_folder, data["name"], "test_fast_prefix_embeddings.pkl")
                self.post_path = os.path.join(data_folder, data["name"], "test_fast_suffix_embeddings.pkl")
        else:
            self.text_path = os.path.join(data_folder, data["name"], "data.txt")
            self.pre_path = os.path.join(data_folder, data["name"], "fast_prefix_embeddings.pkl")
            self.post_path = os.path.join(data_folder, data["name"], "fast_suffix_embeddings.pkl")
        with open(self.text_path, "r") as f:
            self.sents = [sent.strip() for sent in f if sent.strip() != ""]
        with open(self.pre_path, "rb") as f:
            self.pre = pickle.load(f)
        with open(self.post_path, "rb") as f:
            self.post = pickle.load(f)

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        return self.sents[idx], (self.pre[idx], self.post[idx])


class FastCollator:
    def __init__(
        self, 
        data: Dict[str, str], 
        args: argparse.Namespace=None, 
        max_length: int=None, 
        max_context_length_per_k: int=None,
        tokenizer: transformers.PreTrainedTokenizerFast=None
        ):
        '''
        Data collator for FastRetrievalDataset. Aggregate examples in FastRetrievalDataset into batches 
        and provide corresponding attention masks.
        Parameters:
            data: dict, json style dictionary containing the following fields:
                name: str, name of the folder containing the text files.
                ngram: int, frequency of retrieval.
                topk: int, topk embeddings to retrieve for each step.
            max_length: int, maximum length of input tokens. Only valid when args is not given.
            max_context_length_per_k:, int, maximum retrieved embedding length per topk. Only valid when args is not given.
            tokenizer: PreTrainedTokenizerFast, tokenizer of the LM from Huggingface to tokenize sentences. Only valid when args is not given.
        '''
        if args is not None:
            max_length = args.max_length
            max_context_length_per_k = args.max_context_length_per_k
            tokenizer = args.tokenizer
            concat_self = not args.not_concat_self
        self.topk = data["topk"]
        self.ngram = data["ngram"]
        self.max_length = max_length
        self.max_ctx_tok_len = max_context_length_per_k * self.topk
        self.tokenizer = tokenizer
        self.concat_self = concat_self
        if self.concat_self:
            self.mask_gen = AttentionMaskGenerator(self.topk)
        else:
            self.mask_gen = NaiveAttentionMaskGenerator(self.topk)
            
    def __call__(self, batch):
        text_list = []
        pre_list = []
        post_list = []
        for sent, emb in batch:
            pre_list.append(emb[0])
            post_list.append(emb[1])
            text_list.append(sent)
        if self.tokenizer.name_or_path == "gpt2":  # fixing the issue where GPT2Tokenizer does not add special tokens
            inputs = self.tokenizer([self.tokenizer.bos_token + x + self.tokenizer.eos_token for x in text_list], return_tensors="pt", max_length=self.max_length, truncation=True, padding=True, add_special_tokens=False)
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        inputs["ngram"] = self.ngram
        pre_list = pad_sequence(pre_list, batch_first=True)
        post_list = pad_sequence(post_list, batch_first=True)
        pre_list = pre_list[:,:self.max_ctx_tok_len,:]
        post_list = post_list[:,:self.max_ctx_tok_len,:]
        inputs["encoder_hidden_keys"] = pre_list
        inputs["encoder_hidden_values"] = post_list
        mask = self.mask_gen(inputs)
        inputs["encoder_attention_mask"] = mask
        del inputs["ngram"]
        return inputs