import utils
import sys
import argparse
import json
import logging
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
from transformers.utils import (
    logging,
)
from transformers.models.bert import BertLMHeadModel
from train_script import RetrievalGenerationModel, train_function, update_config

logger = logging.get_logger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--external_embedding', action="store_true", default=False)
    parser.add_argument('--pre_path', type=str, default=None)
    parser.add_argument('--post_path', type=str, default=None)
    parser.add_argument('--no_gpu', action="store_true", default=False)
    parser.add_argument('--ngpus', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--from_checkpoint', action="store_true", default=False)
    parser.add_argument('--not_retrieval', action="store_true", default=False)
    parser.add_argument('--not_cross_attention', action="store_true", default=False)
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--max_context_length_per_k', type=int, default=128)
    #parser.add_argument('--nprocs', type=int, default=8)
    #parser.add_argument('--datasets_per_batch', type=int, default=2, help="Number of datasets per batch")
    #parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")
    #parser.add_argument('--no_normalize', action="store_true", default=False, help="If set: Embeddings are not normalized")
    #parser.add_argument('--pooling', default='mean')
    parser.add_argument('--data_folder', default="/data", help="Folder with your dataset files")
    parser.add_argument('--save_head', action="store_true", default=False)
    parser.add_argument('data_config', help="A data_config.json file", default=None)
    parser.add_argument('output')
    args = parser.parse_args()

    # Ensure num proc is devisible by datasets_per_batch
    #assert (args.nprocs % args.datasets_per_batch) == 0
    args.config = transformer.AutoConfig.from_pretrained(args.model_path)
    if not args.from_checkpoint:
        update_config(args)
    args.device = torch.device("cuda") if not args.no_gpu else torch.device("cpu")
    args.device_ids = list(range(args.ngpus))
    logging.info("Output: "+args.output)
    if os.path.exists(args.output):
        print("Output folder already exists.")
        input("Continue?")

    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)

    if args.data_config is None:
        args.data_config = os.path.join(args.data, "data_config.json")
    data_config_path = os.path.join(args.output, 'data_config.json')
    copyfile(args.data_config, data_config_path)

    train_script_path = os.path.join(args.output, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    #Load data config
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)

    #queue = mp.Queue(maxsize=100*args.nprocs)
    assert (args.pre_path is not None and args.post_path is not None) or (args.pre_path is None and args.post_path is None)
    

    args.files = []
    #dataset_indices = []
    args.total_lines = 0

    for data in data_config:
        args.files.append(data)
        #dataset_indices.extend([idx]*data['weight'])
        args.total_lines += data["lines"]

    if args.steps is None:
        args.steps = math.ceil(args.total_lines / args.batch_size)
    args.total_steps = args.steps * args.epochs

    logging.info("Steps per epochs:", args.steps)

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    args.model = RetrievalGenerationModel.from_pretrained(args.model_path, config=args.config)
    args.model.post_post_init(args)
    
    if torch.cuda_device_count() > 1 and not args.no_gpu:
        args.model = DataParallel(args.model, device_ids=list(range(torch.cuda.device_count())))
    
    ### Train Loop
    args.model = args.model.to(args.device)

    args.optimizer = AdamW(params=args.model.parameters(), lr=2e-5, correct_bias=True)

    args.lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_steps,
    )
    train_function(args)
    # Start producer
    #p = mp.Process(target=produce_data, args=(args, queue, filepaths, dataset_indices))
    #p.start()

    # Run training
    #print("Start processes:", args.nprocs)
    print("Training done")
    exit()



# Script was called via:
#python train_many_data_files_v2.py --steps 200000 --batch_size 128 --model nreimers/MiniLM-L6-H384-uncased --max_length_a 64 --max_length_b 250 train_data_configs/multi-qa_v1.json output/multi-qa_v1-MiniLM-L6-mean_cos