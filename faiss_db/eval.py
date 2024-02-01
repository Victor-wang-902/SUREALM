import utils
import sys
import argparse
import json
import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch
import os
from shutil import copyfile
import math
import models
from dataset import load_dataset
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
import warnings
from utils import AttentionMaskGenerator
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions
)

from transformers.models.bert import BertLMHeadModel

import logging
from torch.nn import DataParallel

logging.basicConfig(level=logging.INFO)


def inference(args):
    pass

def evaluate(args):
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)