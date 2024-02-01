import utils
import sys
import argparse
import json
import tqdm
import torch
from torch.utils.data import DataLoader
import torch
import os
from shutil import copyfile
import math
import models
from dataset_dist import load_dataset
import time
from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
import numpy as np
import warnings
from utils import AttentionMaskGenerator, compute_sentence_perplexity
from typing import List, Optional, Tuple, Union
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions
)

from wrappers_roberta import RetrievalGenerationModelRoberta
from wrappers import RetrievalGenerationModel
import logging
from torch.nn import DataParallel
import torch.distributed as dist
logging.basicConfig(level=logging.INFO)


def setup(rank, world_size):    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'    
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


class Handler:
    def __init__(self, args=None, **kwargs):
        if args is None:
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
        self.args = args
        for item in kwargs.items():
            setattr(self.args, item[0], item[1])
    def train(self):
            # Now we train the model
        max_grad_norm = 1
        global_step = 0
        prev_embs = None
        #losses = []
        logging.info("started training")
        for epoch in tqdm.trange(self.args.epochs):
            cur_losses = []
            cur_perplexity = []
            cur_tokens = 0
            total_steps = 0
            for i, data in enumerate(tqdm.tqdm(self.args.files)):
                dataloader = load_dataset(data, self.args)
                if total_steps >= self.args.steps:
                    break
                #start = time.time()
                if not self.args.not_retrieval:
                    if not self.args.external_embedding:
                        pre_path = os.path.join(self.args.data_folder, self.args.files[i]["name"], "prefix_embeddings.pt")
                        post_path = os.path.join(self.args.data_folder, self.args.files[i]["name"], "suffix_embeddings.pt")
                        if prev_embs != pre_path:
                            if self.args.device != torch.device("cpu") and torch.cuda.device_count() > 1:
                                self.args.model.module.load_embeddings(pre_path, post_path)
                            else:
                                self.args.model.load_embeddings(pre_path, post_path)
                            self.args.model.to(self.args.device)
                        prev_embs = pre_path
                    #logging.info(args.model.module.bert.ctx_key_embeddings.embed.weight.requires_grad)
                    #raise Exception()
                #times = []
                #end = time.time()
                #logging.info(end - start)
                #raise Exception()
                #start = time.time()
                #ngram = 1
                #logging.info("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                #logging.info("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                #logging.info("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                for batch in dataloader:

                    #end = time.time()
                    #start = time.time()
                    if total_steps >= self.args.steps:
                        break
                    #start = time.time()
                    self.args.model.train()
                    #if "ngram" in batch.keys():
                    #    ngram = batch["ngram"]
                    #    del batch["ngram"]

                    
                    outputs = self.args.model(**batch.to(self.args.device))
                    #logging.info("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
                    #logging.info("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
                    #logging.info("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
                    #logging.info(torch.cuda.memory_summary())

                    #end = time.time()
                    #times.append(end - start)
                    loss = outputs[0]
                    num_tokens = outputs[-1]
                    #start = time.time()
                    #end = time.time()
                    #logging.info(end-start)
                    #### Get the batch data
                    #batch = queue.get()
                    #print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))

                    # Backward pass
                    #start = time.time()
                    self.args.optimizer.zero_grad()

                    #if self.args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
                    loss.backward(torch.ones_like(loss, dtype=torch.float).to(self.args.device))
                    #else:
                    #    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.args.model.parameters(), max_grad_norm)
                    
                    self.args.optimizer.step()
                    self.args.lr_scheduler.step()
                    total_steps += 1
                    global_step += 1
                    #end = time.time()
                    #times.append(end - start)
                    #Save model
                    
                    detached_loss = torch.sum(loss.detach().cpu())
                    num_tokens = torch.sum(num_tokens.cpu()).item()
                    #print(num_tokens)
                    #raise Exception
                    cur_losses.append(detached_loss.item())
                    cur_tokens += num_tokens
                    if (global_step + 1) % 100 == 0:
                        perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
                        logging.info("current mean train loss "+ str(np.mean(cur_losses)))
                        #perplexity = compute_sentence_perplexity(batch, ngram, args)
                        logging.info("current train perplexity" + str(perplexity))

            mean_loss = np.mean(cur_losses)
            mean_perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
            #losses.append(mean_loss)
            #logging.info("final mean train loss: " + str(mean_loss))
            #logging.info("final train perplexity: " + str(mean_perplexity))
            #logging.info("epoch finished")

            if (epoch + 1) % self.args.save_epochs == 0:
                if not self.args.no_eval:
                    self.evaluate()
                output_path = os.path.join(self.args.output, str(epoch+1))
                logging.info("checkpoint. save model: "+output_path)
                if self.args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
                    self.args.model.module.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
                else:
                    self.args.model.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
                    #end = time.time()
                    #logging.info("time spent"+ str(end - start))
                    #raise Exception()
                
                #end = time.time()
                #logging.info("time spent"+ str(end - start))
                #logging.info(sum(times) / len(times))
                #raise Exception()
                #if args.eval:
            #torch.cuda.empty_cache()

        logging.info("evaluating final model")
        if not self.args.no_eval:
            self.evaluate()
        output_path = os.path.join(self.args.output, "final")
        logging.info("save model final: "+ output_path)
        if self.args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
            self.args.model.module.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
        else:
            self.args.model.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
        

    def evaluate(self):
        prev_embs = None
        #losses = []
        cur_losses = []
        cur_perplexity = []
        cur_tokens = 0
        logging.info("start evaluating")
        for i, data in enumerate(tqdm.tqdm(self.args.files)):
            if self.args.online_eval:
                dataloader = load_dataset(data, self.args, val_exc=True)
            else:
                dataloader = load_dataset(data, self.args, val=True)
            #start = time.time()
            if not self.args.not_retrieval:
                if not self.args.external_embedding:
                    pre_path = os.path.join(self.args.data_folder, self.args.files[i]["name"], "prefix_embeddings.pt")
                    post_path = os.path.join(self.args.data_folder, self.args.files[i]["name"], "suffix_embeddings.pt")
                    if prev_embs != pre_path:
                        self.args.model.module.load_embeddings(pre_path, post_path)
                        self.args.model.to(self.args.device)
                    prev_embs = pre_path
                #logging.info(args.model.module.bert.ctx_key_embeddings.embed.weight.requires_grad)
                #raise Exception()
            #times = []
            #end = time.time()
            #logging.info(end - start)
            #raise Exception()
            #start = time.time()
            #ngram = 1
            for batch in dataloader:
                #end = time.time()
                #start = time.time()
                #start = time.time()
                self.args.model.eval()
                #if "ngram" in batch.keys():
                #    ngram = batch["ngram"]
                #    del batch["ngram"]
                with torch.no_grad():
                    outputs = self.args.model(**batch.to(self.args.device))

                #end = time.time()
                #times.append(end - start)
                loss = outputs[0]
                num_tokens = outputs[-1]
                #print(num_tokens)
                #raise Exception
                logits = outputs[1]
                inds = torch.argmax(logits, dim=-1)
                
                summed_loss = torch.sum(loss.cpu())
                summed_num_tokens = torch.sum(num_tokens.cpu()).item()
                #logging.info(str(num_tokens))
                cur_tokens += summed_num_tokens
                if self.args.debug:
                    words = self.args.tokenizer.batch_decode(inds)
                    expected = self.args.tokenizer.batch_decode(batch.input_ids)
                    perplexity = torch.exp(summed_loss / summed_num_tokens).item()
                    logging.info("predicted: " + str(words))
                    logging.info("expected: " + str(expected))
                    logging.info("batch loss: " + str(loss))
                    logging.info("batch perplexity: " + str(perplexity))

                #start = time.time()
                #end = time.time()
                #logging.info(end-start)
                #### Get the batch data
                #batch = queue.get()
                #print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))

                # Backward pass
                #start = time.time()
                #end = time.time()
                #times.append(end - start)
                #Save model
                

                cur_losses.append(summed_loss.item())
            
        mean_loss = np.mean(cur_losses)
        mean_perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
        #losses.append(mean_loss)
        logging.info("eval mean loss: " + str(mean_loss))
        logging.info("eval perplexity: " + str(mean_perplexity))
        logging.info("evalaution complete")

    def inference(self, prompt):
        pass
        self.args.model.eval()
        if isinstance(prompt, str):
            prompt = [prompt]
        self.args.model.generate()
'''
def train_function(args):
   
    # Now we train the model
    max_grad_norm = 1
    global_step = 0
    prev_embs = None
    #losses = []
    for epoch in tqdm.trange(args.epochs):
        cur_losses = []
        cur_perplexity = []
        total_steps = 0
        for i, data in enumerate(tqdm.tqdm(args.files)):
            dataloader = load_dataset(data, args)
            if total_steps >= args.steps:
                break
            #start = time.time()
            if not args.not_retrieval:
                if not args.external_embedding:
                    pre_path = os.path.join(args.data_folder, args.files[i]["name"], "prefix_embeddings.pt")
                    post_path = os.path.join(args.data_folder, args.files[i]["name"], "suffix_embeddings.pt")
                    if prev_embs != pre_path:
                        args.model.module.load_embeddings(pre_path, post_path)
                        args.model.to(args.device)
                    prev_embs = pre_path
                #logging.info(args.model.module.bert.ctx_key_embeddings.embed.weight.requires_grad)
                #raise Exception()
            #times = []
            #end = time.time()
            #logging.info(end - start)
            #raise Exception()
            #start = time.time()
            #ngram = 1
            for batch in dataloader:
                #end = time.time()
                #start = time.time()
                if total_steps >= args.steps:
                    break
                #start = time.time()
                args.model.train()
                #if "ngram" in batch.keys():
                #    ngram = batch["ngram"]
                #    del batch["ngram"]
                outputs = args.model(**batch.to(args.device))

                #end = time.time()
                #times.append(end - start)
                loss = outputs[0]
                
                #start = time.time()
                #end = time.time()
                #logging.info(end-start)
                #### Get the batch data
                #batch = queue.get()
                #print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))

                # Backward pass
                #start = time.time()
                args.optimizer.zero_grad()
                if args.device == torch.device("cuda"):
                    loss.backward(torch.ones(torch.cuda.device_count()).to(args.device))
                else:
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(args.model.parameters(), max_grad_norm)
                
                args.optimizer.step()
                args.lr_scheduler.step()
                total_steps += 1
                global_step += 1
                #end = time.time()
                #times.append(end - start)
                #Save model
                detached_loss = torch.mean(loss.detach())
                cur_losses.append(detached_loss.item())
                perplexity = torch.exp(detached_loss).item()
                cur_perplexity.append(perplexity)
                if (global_step + 1) % 100 == 0:
                    logging.info("loss "+ str(np.mean(cur_losses)))
                    #perplexity = compute_sentence_perplexity(batch, ngram, args)
                    logging.info("train perplexity" + str(np.mean(cur_perplexity)))

        mean_loss = np.mean(cur_losses)
        mean_perplexity = np.mean(cur_perplexity)
        #losses.append(mean_loss)
        logging.info("mean loss: " + str(mean_loss))
        logging.info("mean perplexity: " + str(mean_perplexity))

        if (epoch + 1) % args.save_epochs == 0:
            if not args.no_eval:
                evaluate(args)
            output_path = os.path.join(args.output, str(epoch+1))
            logging.info("save model: "+output_path)
            if args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
                args.model.module.save_pretrained(output_path, args.save_head, args.save_emb)
            else:
                args.model.save_pretrained(output_path, args.save_head, args.save_emb)
                #end = time.time()
                #logging.info("time spent"+ str(end - start))
                #raise Exception()
            
            #end = time.time()
            #logging.info("time spent"+ str(end - start))
            #logging.info(sum(times) / len(times))
            #raise Exception()
            #if args.eval:

            
    
    output_path = os.path.join(args.output, "final")
    logging.info("save model final: "+ output_path)
    if args.device == torch.device("cuda"):
        args.model.module.save_pretrained(output_path, args.save_head, args.save_emb)
    else:
        args.model.save_pretrained(output_path, args.save_head, args.save_emb)

def evaluate(args):
    prev_embs = None
    #losses = []
    cur_losses = []
    cur_perplexity = []
    for i, data in enumerate(tqdm.tqdm(args.files)):
        dataloader = load_dataset(data, args, val=True)
        #start = time.time()
        if not args.not_retrieval:
            if not args.external_embedding:
                pre_path = os.path.join(args.data_folder, args.files[i]["name"], "prefix_embeddings.pt")
                post_path = os.path.join(args.data_folder, args.files[i]["name"], "suffix_embeddings.pt")
                if prev_embs != pre_path:
                    args.model.module.load_embeddings(pre_path, post_path)
                    args.model.to(args.device)
                prev_embs = pre_path
            #logging.info(args.model.module.bert.ctx_key_embeddings.embed.weight.requires_grad)
            #raise Exception()
        #times = []
        #end = time.time()
        #logging.info(end - start)
        #raise Exception()
        #start = time.time()
        #ngram = 1
        for batch in dataloader:
            #end = time.time()
            #start = time.time()
            #start = time.time()
            args.model.eval()
            #if "ngram" in batch.keys():
            #    ngram = batch["ngram"]
            #    del batch["ngram"]
            with torch.no_grad():
                outputs = args.model(**batch.to(args.device))

            #end = time.time()
            #times.append(end - start)
            loss = outputs[0]
            #start = time.time()
            #end = time.time()
            #logging.info(end-start)
            #### Get the batch data
            #batch = queue.get()
            #print(index, "batch {}x{}".format(len(batch), ",".join([str(len(b)) for b in batch])))

            # Backward pass
            #start = time.time()
            #end = time.time()
            #times.append(end - start)
            #Save model
            avg_loss = torch.mean(loss)
            perplexity = torch.exp(avg_loss).item()
            cur_losses.append(avg_loss.item())
            cur_perplexity.append(perplexity)
        
    mean_loss = np.mean(cur_losses)
    mean_perplexity = np.mean(cur_perplexity)
    #losses.append(mean_loss)
    logging.info("eval mean loss: " + str(mean_loss))
    logging.info("eval mean perplexity: " + str(mean_perplexity))
    '''
 

def update_config_retrieval(args):
    args.has_cross_attention = args.config.add_cross_attention if hasattr(args.config, "add_cross_attention") else False
    args.config.add_cross_attention = not args.not_cross_attention
    args.config.is_retrieval = not args.not_retrieval
    args.config.is_decoder = True
    args.config.bos_token_id = 102
    args.config.eos_token_id = 103
    args.config.max_length = args.max_length
    args.config.max_context_length_per_k = args.max_context_length_per_k
    
def update_config_regular(args):
    args.has_cross_attention = args.config.add_cross_attention if hasattr(args.config, "add_cross_attention") else False
    args.config.add_cross_attention = not args.not_cross_attention
    args.config.is_decoder = True
    args.config.max_length = args.max_length

def main(rank, args):

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--from_scratch', action="store_true", default=False)
    parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--external_embedding', action="store_true", default=False)
    parser.add_argument('--not_fast', action="store_true", default=False)
    parser.add_argument('--pre_path', type=str, default=None)
    parser.add_argument('--post_path', type=str, default=None)
    parser.add_argument('--no_gpu', action="store_true", default=False)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--from_checkpoint', action="store_true", default=False)
    parser.add_argument('--no_eval', action="store_true", default=False)
    parser.add_argument('--eval_only', action="store_true", default=False)
    parser.add_argument('--freeze_backbone', action="store_true", default=False)
    parser.add_argument('--not_retrieval', action="store_true", default=False)
    parser.add_argument('--not_cross_attention', action="store_true", default=False)
    parser.add_argument('--steps', type=int, default=None)
    #parser.add_argument('--save_steps', type=int, default=1000)
    parser.add_argument('--save_epochs', type=int, default=1)
    parser.add_argument('--batch_size_per_gpu', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=128) #generation max length
    parser.add_argument('--max_input', type=int, default=1024) #model input max length
    parser.add_argument('--topk', type=int, default=3) # only valid when --not_fast. for training
    parser.add_argument('--online_eval', action="store_true", default=False)
    parser.add_argument('--eval_topk', type=int, default=3) #only valid when eval_ngram
    parser.add_argument('--eval_ngram', type=int, default=2) #only valid when eval_ngram
    parser.add_argument('--init_cross_attention_weights', action="store_true", default=False)
    parser.add_argument('--max_context_length_per_k', type=int, default=128)
    #parser.add_argument('--nprocs', type=int, default=8)
    #parser.add_argument('--datasets_per_batch', type=int, default=2, help="Number of datasets per batch")
    #parser.add_argument('--scale', type=float, default=20, help="Use 20 for cossim, and 1 when you work with unnormalized embeddings with dot product")
    #parser.add_argument('--no_normalize', action="store_true", default=False, help="If set: Embeddings are not normalized")
    #parser.add_argument('--pooling', default='mean')
    parser.add_argument('--data_folder', default="processed_data", help="Folder with your dataset files")
    parser.add_argument('--save_head', action="store_true", default=False)
    parser.add_argument('--save_emb', action="store_true", default=False)
    parser.add_argument('--data_config', help="A data_config.json file", default=None)
    parser.add_argument('--output', default="retrieval_trained")
    args = parser.parse_args()
    torch.manual_seed(0)
    torch.set_printoptions(threshold=1000000)

    # Ensure num proc is devisible by datasets_per_batch
    #assert (args.nprocs % args.datasets_per_batch) == 0
    args.batch_size = args.batch_size_per_gpu
    args.config = AutoConfig.from_pretrained(args.model_path)
    model_type = args.config.model_type
    if not args.from_checkpoint:
        if not args.not_retrieval:
            update_config_retrieval(args)
        else:
            update_config_regular(args)
    else:

        args.has_cross_attention = not args.not_cross_attention

    args.device = torch.device("cuda") if not args.no_gpu else torch.device("cpu")
    logging.info("Output: "+args.output)
    if os.path.exists(args.output):
        if not args.eval_only:
            raise RuntimeError("Output folder already exists.")

    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)

    if args.data_config is None:
        args.data_config = os.path.join(args.data_folder, "data_config.json")
    else:
        args.data_config = os.path.join(args.data_folder, args.data_config)
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
        #print(data)
        #dataset_indices.extend([idx]*data['weight'])
        args.total_lines += data["lines"]

    if args.steps is None:
        args.steps = math.ceil(args.total_lines / args.batch_size)
    args.total_steps = args.steps * args.epochs

    logging.info("Steps per epochs:" + str(args.steps))
    logging.info("Total steps:" + str(args.total_steps))

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.tokenizer.pad_token is None:
        args.tokenizer.pad_token = args.tokenizer.eos_token
    is_baseline = False
    if model_type == "bert":
        model_class = RetrievalGenerationModel
    elif model_type == "roberta":
        model_class = RetrievalGenerationModelRoberta
    else:
        model_class = AutoModelForCausalLM
        is_baseline = True


    if args.from_scratch:
        try:
            args.model = model_class(config=args.config, args=args)
        except:
            args.model = model_class.from_config(config=args.config)
    else:
        try:
            args.model = model_class.from_pretrained(args.model_path, config=args.config, args=args)
        except:
            args.model = model_class.from_pretrained(args.model_path, config=args.config)
    if not is_baseline:
        args.model.post_post_init(args)
    #print(args.model)
    if torch.cuda.device_count() > 1 and not args.no_gpu:
        args.model = DataParallel(args.model, device_ids=list(range(torch.cuda.device_count())))
    
    ### Train Loop
    args.model = args.model.to(args.device)

    args.optimizer = AdamW(params=args.model.parameters(), lr=2e-5, correct_bias=True)

    args.lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=args.optimizer,
        num_warmup_steps=500,
        num_training_steps=args.total_steps,
    )
    handler = Handler(args=args)
    if args.eval_only:
        handler.evaluate()
    else:
        handler.train()
    #train_function(args)
    # Start producer
    #p = mp.Process(target=produce_data, args=(args, queue, filepaths, dataset_indices))
    #p.start()

    # Run training
    #print("Start processes:", args.nprocs)
    print("Training done")
    exit()
