import os
import argparse
from torch.utils.data import Dataset
import transformers


class TextDataset(Dataset):
    '''
    Dataset for regular text data. Used for baseline.
    '''
    def __init__(self, data, args=None, val=False, data_folder=None):
        if args is not None:
            data_folder = args.data_folder
        if val:
            if not args.eval_only:
                self.text_path = os.path.join(data_folder, data["name"], "data_eval.txt")
            else:
                self.text_path = os.path.join(data_folder, data["name"], "data_test.txt")
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
    '''
    Data collator for regular text data. Used for baseline.
    '''
    def __init__(self, args=None, max_length=None, tokenizer=None):
        if args is not None:
            max_length = args.max_length
            tokenizer = args.tokenizer
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __call__(self, batch):
        text_list = []
        for sent in batch:
            text_list.append(sent)
        if self.tokenizer.name_or_path == "gpt2":  # fixing the issue where GPT2Tokenizer does not add special tokens
            inputs = self.tokenizer([tok.bos_token + x + tok_eos_token for x in text_list], return_tensors="pt", max_length=self.max_length, truncation=True, padding=True, add_special_tokens=False)
        else:
            inputs = self.tokenizer(text_list, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True)
        return inputs
