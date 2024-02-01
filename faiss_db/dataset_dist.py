import json
import torch
from torch.utils.data import Dataset, DataLoader
import os
import faiss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from utils import pad_emb, AttentionMaskGenerator
import pickle
import numpy as np
import logging
import time
from fast_encoder import skip_encode


def extract_dstc_knowledge(filepath, outpath):
    with open(filepath, "r") as f:
        entries = json.load(f)
    db = []
    for cat in entries:
        for item in entries[cat]:
            for docs in entries[cat][item]["docs"]:
                db.append(entries[cat][item]["docs"][docs]["title"] + "\n")
                db.append(entries[cat][item]["docs"][docs]["body"] + "\n")
    with open(outpath, "w") as f:
        f.writelines(db)

def extract_dstc(filepath, outpath):
    with open(filepath, "r") as f:
        entries = json.load(f)
    db = []
    for item in entries:
        for entry in item:
            db.append(entry["text"] + "\n")
    db = set(db)
    with open(outpath, "w") as f:
        f.writelines(db)

class FastRetrievalDataset(Dataset):
    def __init__(self, data, args=None, val=False, data_folder="fast_processed_data", external_embedding=False, device=torch.device("cpu")):
        if args is not None:
            data_folder = args.data_folder
            external_embedding = args.external_embedding
            device = args.device
        self.external_embedding = external_embedding
        self.val = val
        self.ngram = data["ngram"]
        self.topk = data["topk"]
        if self.val:
            self.text_path = os.path.join(data_folder, data["name"], "data_eval.txt")
            if external_embedding:
                self.pre_path = os.path.join(data_folder, data["name"], "val_fast_prefix_embeddings.pkl")
                self.post_path = os.path.join(data_folder, data["name"], "val_fast_suffix_embeddings.pkl")
            else:
                self.ind_path = os.path.join(data_folder, data["name"], "val_fast_indices.pkl")
        else:
            self.text_path = os.path.join(data_folder, data["name"], "data.txt")
            if external_embedding:
                self.pre_path = os.path.join(data_folder, data["name"], "fast_prefix_embeddings.pkl")
                self.post_path = os.path.join(data_folder, data["name"], "fast_suffix_embeddings.pkl")
            else:
                self.ind_path = os.path.join(data_folder, data["name"], "fast_indices.pkl")
        with open(self.text_path, "r") as f:
            self.sents = [sent.strip() for sent in f if sent.strip() != ""]
        if self.external_embedding:
            with open(self.pre_path, "rb") as f:
                self.pre = pickle.load(f)
            with open(self.post_path, "rb") as f:
                self.post = pickle.load(f)
        else:
            with open(self.ind_path, "rb") as f:
                self.ind = pickle.load(f)
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        if self.external_embedding:
            return self.sents[idx], (self.pre[idx], self.post[idx])
        else:
            return self.sents[idx], self.ind[idx]

class FastCollator:
    def __init__(self, data, args=None, max_length=128, max_context_length_per_k=128, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        if args is not None:
            max_length = args.max_length
            max_context_length_per_k = args.max_context_length_per_k
            tokenizer = args.tokenizer
        self.topk = data["topk"]
        self.ngram = data["ngram"]
        self.max_length = max_length
        self.max_ctx_tok_len = max_context_length_per_k * self.topk
        self.tokenizer = tokenizer
        self.mask_gen = AttentionMaskGenerator(self.topk)
        #self.vocab = self.build_vocab(args)

    #def build_vocab(self, args):
    #    def yield_tokens():
    #        for text, _ in args.data_iter:
    #            yield args.tokenizer(text)
    #    vocab = build_vocab_from_iterator(yield_tokens(), specials=["<sos>", "<eos>", "<unk>", "<pad>"])
    #    vocab.set_default_index(vocab["<unk>"])
    #    return vocab

    #def text_pipeline(self, text, batch_sent_max_len):
    #    max_len = batch_sent_max_len if batch_sent_max_len < self.max_txt_tok_len else self.max_txt_tok_len
    #    word_list = self.vocab(self.tokenizer(text))
    #    word_list = word_list[:self.max_len-2]
    #    word_list = [self.vocab["<sos>"]] + word_list + [self.vocab["<eos>"]]
    #    word_list = word_list + (self.max_len - len(word_list)) * [self.vocab["<pad>"]]
    #    word_list = torch.tensor(word_list, dtype=torch.int64)
    #    return word_list
        
    #def text_pipeline(self, text):
    #    inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
    #    return inputs 
    #def context_pipeline(self, indices, batch_ctx_max_len):
    #    max_len = batch_ctx_max_len if batch_ctx_max_len < self.max_ctx_tok_len else self.max_ctx_tok_len
    #    pre_embeds = [self.faiss_index.reconstruct(i) for i in indices]
    #    pre_embeds = np.stack(normalize(pre_embeds), dim=0)
    #    topk_indices = self.faiss_index.search(embed) ##modify here if topk needs an projection
    #    topk_indices = torch.tensor(topk_indices, dtype=torch.int64).reshape(-1)
    #    topk_indices = torch.cat((topk_indices[:max_len], torch.tensor((max_len - topk_indices.shape[0]) * [0], dtype=torch.int64)))
    #    return topk_indices

    def __call__(self, batch):
        text_list = []
        idx_list = []
        pre_list = []
        post_list = []
        #for sent, indices in batch:
        #    sent_len = len(self.vocab(self.tokenizer(sent)))
        #    ind_len = len(indices)
        #    batch_sent_max_len = sent_len if sent_len > batch_sent_max_len else batch_sent_max_len
        #    batch_ctx_max_len = ind_len if ind_len > batch_ctx_max_len else batch_ctx_max_len
        #append_start = time.time()
        for sent, emb in batch:

            if isinstance(emb, tuple):
                pre_list.append(emb[0])
                post_list.append(emb[1])
            else:
                idx_list.append(emb)
        #    processed_text = self.text_pipeline(sent, batch_sent_max_len)
            text_list.append(sent)
            #topk_indices = context_pipeline(indices, batch_ctx_max_len)
        #append_end = time.time()
        #tok_start = time.time()
        inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        #tok_end = time.time()
        inputs["ngram"] = self.ngram
        #pad_start = time.time()
        if len(idx_list) != 0:
            idx_list = pad_sequence(idx_list, batch_first=True)
            idx_list = idx_list[:,:self.max_ctx_tok_len]
            inputs["encoder_indices"] = idx_list
        
        else:
            pre_list = pad_sequence(pre_list, batch_first=True)
            post_list = pad_sequence(post_list, batch_first=True)
            pre_list = pre_list[:,:self.max_ctx_tok_len,:]
            post_list = post_list[:,:self.max_ctx_tok_len,:]
            inputs["encoder_hidden_keys"] = pre_list
            inputs["encoder_hidden_values"] = post_list
        #pad_end = time.time()
        #msk_start = time.time()
        mask = self.mask_gen(inputs)
        #msk_end = time.time()
        inputs["encoder_attention_mask"] = mask
        del inputs["ngram"]
        #logging.info("apd"+str(append_end - append_start))
        #logging.info("tok"+str(tok_end - tok_start))
        #logging.info("pad"+str(pad_end - pad_start))
        #logging.info("msk"+str(msk_end - msk_start))
        #logging.info("total"+str(msk_end - append_start))

        #raise Exception()

        return inputs

class RetrievalValDataset(Dataset):
    def __init__(self, data, args=None, val=False, model=AutoModel.from_pretrained("multi-qa-MiniLM-L6-cos-v1"), tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased"), data_folder="processed_data", topk=3, ngram=2, external_embedding=False, device=torch.device("cpu")):
        if args is not None:
            data_folder = args.data_folder
            topk = args.eval_topk
            ngram = args.eval_ngram
            external_embedding = args.external_embedding
            device = args.device
            tokenizer = args.tokenizer
        self.text_path = os.path.join(data_folder, data["name"], "data_eval.txt")
        #self.mapping_path = os.path.join(data_folder, data["name"], "val_embedding_mappings.pkl")
        #self.val_prefix_path = os.path.join(data_folder, data["name"], "val_prefix_embeddings.pt")
        self.model = model

        self.index_path = os.path.join(data_folder, data["name"], "prefix_index.index")
        
        self.prefix_path = os.path.join(data_folder, data["name"], "prefix_embeddings.pt")
        self.suffix_path = os.path.join(data_folder, data["name"], "suffix_embeddings.pt")
        self.ngram = ngram
        self.topk = topk
        self.tokenizer = tokenizer
        self.external_embedding = external_embedding
        with open(self.text_path, "r") as f:
            self.sents = [sent.strip() for sent in f if sent.strip() != ""]
        
        self.index = faiss.read_index(self.index_path)
        #print(self.index)
        if device != torch.device("cpu"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        #print(self.index)
        
        self.pre = torch.load(self.prefix_path)
        if external_embedding:
            self.post = torch.load(self.suffix_path)
        #self.embedding = torch.load(os.path.join(self.index_folder, "suffix_embeddings.pt"))

    def get_index(self):
        return self.index
    
    def get_prefix(self):
        return self.pre

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent = self.sents[idx]
        pre_embeds, _, _ = skip_encode(sent, self.ngram, tokenizer=self.tokenizer, model=self.model)
        if len(pre_embeds) == 0:
            if self.external_embedding:
                return sent, (self.pre(torch.tensor([0])), self.post(torch.tensor([0]))), self.ngram
            else:
                return sent, torch.tensor([0]), self.ngram
        
        pre_embeds = torch.stack(pre_embeds).numpy()
        #print(pre_embeds)
        #print(sent)
        #pre_embeds = self.pre(torch.tensor(embs)).numpy()
        _, topk_indices = self.index.search(pre_embeds, self.topk) ##modify here if topk needs an projection
        #end = time.time()
        topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        if self.external_embedding:
            keys = self.pre(topk_indices)
            values = self.post(topk_indices)
            return sent, (keys, values), self.ngram
        
        #logging.info("timespent" + str(end - start))
        return sent, topk_indices, self.ngram

class RetrievalDataset(Dataset):
    def __init__(self, data, args=None, val=False, data_folder="processed_data", topk=3, external_embedding=False, device=torch.device("cpu")):
        if args is not None:
            data_folder = args.data_folder
            topk = args.topk
            external_embedding = args.external_embedding
            device = args.device
        self.val = val
        if self.val:
            self.text_path = os.path.join(data_folder, data["name"], "data_eval.txt")
            self.mapping_path = os.path.join(data_folder, data["name"], "val_embedding_mappings.pkl")
            self.val_prefix_path = os.path.join(data_folder, data["name"], "val_prefix_embeddings.pt")
 
        else:
            self.text_path = os.path.join(data_folder, data["name"], "data.txt")
            self.mapping_path = os.path.join(data_folder, data["name"], "embedding_mappings.pkl")

        self.index_path = os.path.join(data_folder, data["name"], "prefix_index.index")
        
        self.prefix_path = os.path.join(data_folder, data["name"], "prefix_embeddings.pt")
        self.suffix_path = os.path.join(data_folder, data["name"], "suffix_embeddings.pt")
        self.ngram = data["ngram"]
        self.topk = topk
        self.external_embedding = external_embedding
        with open(self.text_path, "r") as f:
            self.sents = [sent.strip() for sent in f if sent.strip() != ""]
        
        with open(self.mapping_path, "rb") as f:
            self.emb_map = pickle.load(f)
        self.index = faiss.read_index(self.index_path)
        #print(self.index)
        if device != torch.device("cpu"):
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        #print(self.index)
        if val:
            self.val_pre = torch.load(self.val_prefix_path)
        self.pre = torch.load(self.prefix_path)
        if external_embedding:
            self.post = torch.load(self.suffix_path)
        #self.embedding = torch.load(os.path.join(self.index_folder, "suffix_embeddings.pt"))

    def get_index(self):
        return self.index
    
    def get_prefix(self):
        return self.pre

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        
        sent = self.sents[idx]
        embs = self.emb_map[idx]
        if len(embs) == 0:
            if self.external_embedding:
                return sent, (self.pre(torch.tensor([0])), self.post(torch.tensor([0]))), self.ngram
            else:
                return sent, torch.tensor([0]), self.ngram
        if self.val:
            pre_embeds = self.val_pre(torch.tensor(embs)).numpy()
        else:
            pre_embeds = self.pre(torch.tensor(embs)).numpy()
        #pre_embeds = np.stack(pre_embeds, axis=0)
        #start = time.time()
        if self.val:   
            _, topk_indices = self.index.search(pre_embeds, self.topk) ##modify here if topk needs an projection
        else:
            _, topk_indices = self.index.search(pre_embeds, len(embs) + self.topk) ##modify here if topk needs an projection
            topk_indices = np.array([np.setdiff1d(topk_indices[i:], embs)[:self.topk] for i in range(topk_indices.shape[0])])
        #end = time.time()
        topk_indices = topk_indices.reshape(-1)
        topk_indices = torch.tensor(topk_indices, dtype=torch.int32)
        if self.external_embedding:
            keys = self.pre(topk_indices)
            values = self.post(topk_indices)
            return sent, (keys, values), self.ngram
        
        #logging.info("timespent" + str(end - start))
        return sent, topk_indices, self.ngram


class Collator:
    def __init__(self, args=None, max_length=128, max_context_length_per_k=128, topk=3, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        if args is not None:
            max_length = args.max_length
            max_context_length_per_k = args.max_context_length_per_k
            topk = args.topk
            print(topk)
            raise Exception
            tokenizer = args.tokenizer

        self.max_length = max_length
        self.max_ctx_tok_len = max_context_length_per_k * topk
        self.tokenizer = tokenizer
        self.mask_gen = AttentionMaskGenerator(topk)
        #self.vocab = self.build_vocab(args)

    #def build_vocab(self, args):
    #    def yield_tokens():
    #        for text, _ in args.data_iter:
    #            yield args.tokenizer(text)
    #    vocab = build_vocab_from_iterator(yield_tokens(), specials=["<sos>", "<eos>", "<unk>", "<pad>"])
    #    vocab.set_default_index(vocab["<unk>"])
    #    return vocab

    #def text_pipeline(self, text, batch_sent_max_len):
    #    max_len = batch_sent_max_len if batch_sent_max_len < self.max_txt_tok_len else self.max_txt_tok_len
    #    word_list = self.vocab(self.tokenizer(text))
    #    word_list = word_list[:self.max_len-2]
    #    word_list = [self.vocab["<sos>"]] + word_list + [self.vocab["<eos>"]]
    #    word_list = word_list + (self.max_len - len(word_list)) * [self.vocab["<pad>"]]
    #    word_list = torch.tensor(word_list, dtype=torch.int64)
    #    return word_list
        
    #def text_pipeline(self, text):
    #    inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
    #    return inputs 
    #def context_pipeline(self, indices, batch_ctx_max_len):
    #    max_len = batch_ctx_max_len if batch_ctx_max_len < self.max_ctx_tok_len else self.max_ctx_tok_len
    #    pre_embeds = [self.faiss_index.reconstruct(i) for i in indices]
    #    pre_embeds = np.stack(normalize(pre_embeds), dim=0)
    #    topk_indices = self.faiss_index.search(embed) ##modify here if topk needs an projection
    #    topk_indices = torch.tensor(topk_indices, dtype=torch.int64).reshape(-1)
    #    topk_indices = torch.cat((topk_indices[:max_len], torch.tensor((max_len - topk_indices.shape[0]) * [0], dtype=torch.int64)))
    #    return topk_indices

    def __call__(self, batch):
        text_list = []
        idx_list = []
        pre_list = []
        post_list = []
        #for sent, indices in batch:
        #    sent_len = len(self.vocab(self.tokenizer(sent)))
        #    ind_len = len(indices)
        #    batch_sent_max_len = sent_len if sent_len > batch_sent_max_len else batch_sent_max_len
        #    batch_ctx_max_len = ind_len if ind_len > batch_ctx_max_len else batch_ctx_max_len
        #append_start = time.time()
        for sent, emb, ngram in batch:

            if isinstance(emb, tuple):
                pre_list.append(emb[0])
                post_list.append(emb[1])
            else:
                idx_list.append(emb)
        #    processed_text = self.text_pipeline(sent, batch_sent_max_len)
            text_list.append(sent)
            #topk_indices = context_pipeline(indices, batch_ctx_max_len)
        #append_end = time.time()
        #tok_start = time.time()
        inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        #tok_end = time.time()
        inputs["ngram"] = ngram
        #pad_start = time.time()
        if len(idx_list) != 0:
            idx_list = pad_sequence(idx_list, batch_first=True)
            idx_list = idx_list[:,:self.max_ctx_tok_len]
            inputs["encoder_indices"] = idx_list
        
        else:
            pre_list = pad_sequence(pre_list, batch_first=True)
            post_list = pad_sequence(post_list, batch_first=True)
            pre_list = pre_list[:,:self.max_ctx_tok_len,:]
            post_list = post_list[:,:self.max_ctx_tok_len,:]
            inputs["encoder_hidden_keys"] = pre_list
            inputs["encoder_hidden_values"] = post_list
        #pad_end = time.time()
        #msk_start = time.time()
        mask = self.mask_gen(inputs)
        #msk_end = time.time()
        inputs["encoder_attention_mask"] = mask
        print("mask",mask)
        print("encoder shape", inputs["encoder_hidden_keys"].shape)
        print("input ids", inputs["input_ids"])
        del inputs["ngram"]
        #logging.info("apd"+str(append_end - append_start))
        #logging.info("tok"+str(tok_end - tok_start))
        #logging.info("pad"+str(pad_end - pad_start))
        #logging.info("msk"+str(msk_end - msk_start))
        #logging.info("total"+str(msk_end - append_start))

        #raise Exception()

        return inputs

class ValCollator:
    def __init__(self, args=None, max_length=128, max_context_length_per_k=128, eval_topk=3, tokenizer=AutoTokenizer.from_pretrained("bert-base-uncased")):
        if args is not None:
            max_length = args.max_length
            max_context_length_per_k = args.max_context_length_per_k
            topk = args.eval_topk
            tokenizer = args.tokenizer

        self.max_length = max_length
        self.max_ctx_tok_len = max_context_length_per_k * topk
        self.tokenizer = tokenizer
        self.mask_gen = AttentionMaskGenerator(topk)
        #self.vocab = self.build_vocab(args)

    #def build_vocab(self, args):
    #    def yield_tokens():
    #        for text, _ in args.data_iter:
    #            yield args.tokenizer(text)
    #    vocab = build_vocab_from_iterator(yield_tokens(), specials=["<sos>", "<eos>", "<unk>", "<pad>"])
    #    vocab.set_default_index(vocab["<unk>"])
    #    return vocab

    #def text_pipeline(self, text, batch_sent_max_len):
    #    max_len = batch_sent_max_len if batch_sent_max_len < self.max_txt_tok_len else self.max_txt_tok_len
    #    word_list = self.vocab(self.tokenizer(text))
    #    word_list = word_list[:self.max_len-2]
    #    word_list = [self.vocab["<sos>"]] + word_list + [self.vocab["<eos>"]]
    #    word_list = word_list + (self.max_len - len(word_list)) * [self.vocab["<pad>"]]
    #    word_list = torch.tensor(word_list, dtype=torch.int64)
    #    return word_list
        
    #def text_pipeline(self, text):
    #    inputs = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
    #    return inputs 
    #def context_pipeline(self, indices, batch_ctx_max_len):
    #    max_len = batch_ctx_max_len if batch_ctx_max_len < self.max_ctx_tok_len else self.max_ctx_tok_len
    #    pre_embeds = [self.faiss_index.reconstruct(i) for i in indices]
    #    pre_embeds = np.stack(normalize(pre_embeds), dim=0)
    #    topk_indices = self.faiss_index.search(embed) ##modify here if topk needs an projection
    #    topk_indices = torch.tensor(topk_indices, dtype=torch.int64).reshape(-1)
    #    topk_indices = torch.cat((topk_indices[:max_len], torch.tensor((max_len - topk_indices.shape[0]) * [0], dtype=torch.int64)))
    #    return topk_indices

    def __call__(self, batch):
        text_list = []
        idx_list = []
        pre_list = []
        post_list = []
        #for sent, indices in batch:
        #    sent_len = len(self.vocab(self.tokenizer(sent)))
        #    ind_len = len(indices)
        #    batch_sent_max_len = sent_len if sent_len > batch_sent_max_len else batch_sent_max_len
        #    batch_ctx_max_len = ind_len if ind_len > batch_ctx_max_len else batch_ctx_max_len
        #append_start = time.time()
        for sent, emb, ngram in batch:

            if isinstance(emb, tuple):
                pre_list.append(emb[0])
                post_list.append(emb[1])
            else:
                idx_list.append(emb)
        #    processed_text = self.text_pipeline(sent, batch_sent_max_len)
            text_list.append(sent)
            #topk_indices = context_pipeline(indices, batch_ctx_max_len)
        #append_end = time.time()
        #tok_start = time.time()
        inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        #tok_end = time.time()
        inputs["ngram"] = ngram
        #pad_start = time.time()
        if len(idx_list) != 0:
            idx_list = pad_sequence(idx_list, batch_first=True)
            idx_list = idx_list[:,:self.max_ctx_tok_len]
            inputs["encoder_indices"] = idx_list
        
        else:
            pre_list = pad_sequence(pre_list, batch_first=True)
            post_list = pad_sequence(post_list, batch_first=True)
            pre_list = pre_list[:,:self.max_ctx_tok_len,:]
            post_list = post_list[:,:self.max_ctx_tok_len,:]
            inputs["encoder_hidden_keys"] = pre_list
            inputs["encoder_hidden_values"] = post_list
        #pad_end = time.time()
        #msk_start = time.time()
        mask = self.mask_gen(inputs)
        #msk_end = time.time()
        inputs["encoder_attention_mask"] = mask
        print("mask",mask)
        print("encoder shape", inputs["encoder_hidden_keys"].shape)
        print("input ids", inputs["input_ids"])
        del inputs["ngram"]
        #logging.info("apd"+str(append_end - append_start))
        #logging.info("tok"+str(tok_end - tok_start))
        #logging.info("pad"+str(pad_end - pad_start))
        #logging.info("msk"+str(msk_end - msk_start))
        #logging.info("total"+str(msk_end - append_start))

        #raise Exception()

        return inputs

class TextDataset(Dataset):
    def __init__(self, data, args=None, val=False, data_folder="processed_data"):
        if args is not None:
            data_folder = args.data_folder
        if val:
            self.text_path = os.path.join(data_folder, data["name"], "data_eval.txt")
        else:
            self.text_path = os.path.join(data_folder, data["name"], "data.txt")
        with open(self.text_path, "r") as f:
            self.sents = [sent.strip() for sent in f if sent.strip() != ""]
        
    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent = self.sents[idx]
        return sent


class TextCollator:
    def __init__(self, args=None, max_length=128, tokenizer=None):
        if args is not None:
            max_length = args.max_length
            tokenizer = args.tokenizer

        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        #start = time.time()
        text_list = []

        for sent in batch:
            text_list.append(sent)
        inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        #end = time.time()
        #logging.info("total"+str(end-start))
        return inputs



def load_dataset(data, args, val=False, val_exc=False):
    if args.not_retrieval:
        dataset_class = TextDataset
        collator = TextCollator(args)
    elif val_exc:
        dataset_class = RetrievalValDataset
        collator = ValCollator(args)
        val = True
    elif not args.not_fast:
        dataset_class = FastRetrievalDataset
        collator = FastCollator(data, args)
    else:
        dataset_class = RetrievalDataset
        collator = Collator(args)
    dataset = dataset_class(data, args=args, val=val)
    shuffle = True if not val else False
    #args.data_iter = iter(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collator)
    return dataloader
    '''for data in args.files:
        #index_path = os.path.join(args.data_folder, data["name"], "prefix_index.index")
        #index = faiss.read_index(index_path)
        #print(self.index)
        #if args.device != torch.device("cpu"):
        #    res = faiss.StandardGpuResources()
        #    index = faiss.index_cpu_to_gpu(res, 0, index)
        dataset = dataset_class(data, args)
        #args.data_iter = iter(dataset)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
        yield dataloader'''
    
def load_val_dataset(data, args):
    if args.not_retrieval:
        dataset_class = TextDataset
        collator = TextCollator(args)
    else:
        dataset_class = RetrievalDataset
        collator = Collator(args)
    dataset = dataset_class(data, args)
    #args.data_iter = iter(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    return dataloader


if __name__ == "__main__":
    extract_dstc("alexa-with-dstc9-track1-dataset/data_eval/test/logs.json",
                    "data_eval.txt")
