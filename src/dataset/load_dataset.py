import argparse
import torch
from torch.utils.data import DataLoader

from .retrieval_dataset import FastRetrievalDataset, FastCollator
from .text_dataset import TextDataset, TextCollator

def load_dataset(data, args, val=False):
    if args.not_retrieval:
        dataset_class = TextDataset
        collator = TextCollator(args)
    else:
        dataset_class = FastRetrievalDataset
        collator = FastCollator(data, args)
    dataset = dataset_class(data, args=args, val=val)
    shuffle = True if not val else False
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collator)
    return dataloader