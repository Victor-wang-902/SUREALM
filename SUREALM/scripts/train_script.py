import sys
import argparse
import json
import tqdm
import logging

import os
from shutil import copyfile
import math

from transformers import (
    AdamW,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    get_linear_schedule_with_warmup,
)

import numpy as np
import torch
from torch import nn

from ..src.dataset import load_dataset
from ..src.wrappers import RetrievalGenerationModelGPT2
from ..src.wrappers import RetrievalGenerationModelRoberta
from ..src.wrappers import RetrievalGenerationModel

logging.basicConfig(level=logging.INFO)


'''
Adapted from the training procedure used by sBERT on Huggingface.
'''


class Handler:
    '''
    Driver code for the main training loop as well as evaluation method.
    '''
    def __init__(self, args=None, **kwargs):
        if args is None:
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
        self.args = args
        for item in kwargs.items():
            setattr(self.args, item[0], item[1])

    def train(self):
        # training loop
        max_grad_norm = 1
        global_step = 0
        prev_embs = None
        logging.info("started training")
        for epoch in tqdm.trange(self.args.epochs):
            cur_losses = []
            cur_perplexity = []
            cur_tokens = 0
            total_steps = 0
            for i, data in enumerate(tqdm.tqdm(self.args.files)):  # TODO: the outer loop was for multi file training, remove for faster single file loading between epochs.
                dataloader = load_dataset(data, self.args)
                if total_steps >= self.args.steps:
                    break
                for batch in dataloader:
                    if total_steps >= self.args.steps:
                        break
                    self.args.model.train()                    
                    outputs = self.args.model(**batch.to(self.args.device))
                    loss = outputs[0]
                    num_tokens = outputs[-1]

                    self.args.optimizer.zero_grad()
                    loss.backward(torch.ones_like(loss, dtype=torch.float).to(self.args.device))
                    torch.nn.utils.clip_grad_norm_(self.args.model.parameters(), max_grad_norm)
                    
                    self.args.optimizer.step()
                    self.args.lr_scheduler.step()
                    total_steps += 1
                    global_step += 1
                    
                    detached_loss = torch.sum(loss.detach().cpu())  # saving loss for display
                    num_tokens = torch.sum(num_tokens.cpu()).item()

                    cur_losses.append(detached_loss.item())
                    cur_tokens += num_tokens
                    if (global_step + 1) % 100 == 0:  # report every 100 steps
                        perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
                        logging.info("current mean train loss "+ str(np.mean(cur_losses)))
                        logging.info("current train perplexity" + str(perplexity))

            mean_loss = np.mean(cur_losses)
            mean_perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
            logging.info("final mean train loss: " + str(mean_loss))
            logging.info("final train perplexity: " + str(mean_perplexity))
            logging.info("epoch finished")

            if (epoch + 1) % self.args.save_epochs == 0:  # save model
                if not self.args.no_eval:
                    self.evaluate()
                    if args.test_eval:  # hack to evaluate on test set too.
                        self.evaluate(test=True)
                output_path = os.path.join(self.args.output, str(epoch+1))
                logging.info("checkpoint. save model: "+output_path)
                if self.args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
                    self.args.model.module.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
                else:
                    self.args.model.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
            del dataloader  # hack to prevent RAM error

        logging.info("evaluating final model")
        if not self.args.no_eval:
            self.evaluate()
            if args.test_eval:
                self.evaluate(test=True)
        output_path = os.path.join(self.args.output, "final")
        logging.info("save model final: "+ output_path)
        if self.args.device == torch.device("cuda") and torch.cuda.device_count() > 1:
            self.args.model.module.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
        else:
            self.args.model.save_pretrained(output_path, self.args.save_head, self.args.save_emb)
        

    def evaluate(self, test=False):
        # evaluation loop
        prev_embs = None
        cur_losses = []
        cur_perplexity = []
        cur_tokens = 0
        to_eval = "test" if test else "validation"
        logging.info("start evaluating on " + to_eval)
        for i, data in enumerate(tqdm.tqdm(self.args.files)):
            dataloader = load_dataset(data, self.args, val=True, test=test)
            for batch in dataloader:
                self.args.model.eval()
                with torch.no_grad():
                    outputs = self.args.model(**batch.to(self.args.device))
                loss = outputs[0]
                num_tokens = outputs[-1]
                logits = outputs[1]
                inds = torch.argmax(logits, dim=-1)
                
                summed_loss = torch.sum(loss.cpu())
                summed_num_tokens = torch.sum(num_tokens.cpu()).item()
                cur_tokens += summed_num_tokens
                if self.args.debug:  # analyze perplexity and monitor predictions sentence by sentence
                    words = self.args.tokenizer.batch_decode(inds)
                    expected = self.args.tokenizer.batch_decode(batch.input_ids)
                    perplexity = torch.exp(summed_loss / summed_num_tokens).item()
                    logging.info("predicted: " + str(words))
                    logging.info("expected: " + str(expected))
                    logging.info("batch loss: " + str(loss))
                    logging.info("batch perplexity: " + str(perplexity))

                cur_losses.append(summed_loss.item())
            
        mean_loss = np.mean(cur_losses)
        mean_perplexity = torch.exp(torch.sum(torch.tensor(cur_losses)) / cur_tokens).item()
        logging.info("eval mean loss: " + str(mean_loss))
        logging.info("eval perplexity: " + str(mean_perplexity))
        logging.info("evalaution complete")


def update_config_retrieval(args):
    '''
    First time initializing a SUREALM from a pretrained model
    '''
    args.has_cross_attention = args.config.add_cross_attention if hasattr(args.config, "add_cross_attention") else False
    args.config.add_cross_attention = not args.not_cross_attention
    args.config.is_retrieval = not args.not_retrieval
    args.config.is_decoder = True

    
def update_config_regular(args):
    '''
    First time initializing a baseline model from a pretrained model
    '''
    args.has_cross_attention = args.config.add_cross_attention if hasattr(args.config, "add_cross_attention") else False
    args.config.add_cross_attention = not args.not_cross_attention
    args.config.is_decoder = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", default=False)
    parser.add_argument('--from_scratch', action="store_true", default=False, help="whether to train the model from scratch")
    parser.add_argument('--model_path', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help="path or name of the model to initialize weights and architecture with")
    parser.add_argument('--no_gpu', action="store_true", default=False, help="do not use GPU for training")
    parser.add_argument('--epochs', type=int, default=1, help="number of epochs to train for")
    parser.add_argument('--from_checkpoint', action="store_true", default=False, help="whether model_path is a checkpoint or not")
    parser.add_argument('--no_eval', action="store_true", default=False, help="skip evaluation")
    parser.add_argument('--test_eval', action="store_true", default=False, help="also evaluate on test data when training given test data is available")
    parser.add_argument('--eval_only', action="store_true", default=False, help="only do evaluation on the test data specified")
    parser.add_argument('--freeze_backbone', action="store_true", default=False, help="whether to freeze backbone weights and only train LM head")
    parser.add_argument('--not_retrieval', action="store_true", default=False, help="disable retrieval. set to True if training a baseline")
    parser.add_argument('--not_cross_attention', action="store_true", default=False, help="disable cross attention block. set to True if training a baseline")
    parser.add_argument('--steps', type=int, default=None, help="number of steps per epoch")
    parser.add_argument('--lr', type=float, default=2e-5, help="initial learning rate")
    parser.add_argument('--save_epochs', type=int, default=1, help="save every number of epochs")
    parser.add_argument('--batch_size_per_gpu', type=int, default=64, help="batch size for each GPU. if only CPU is available, then this argument determines the total batch size")
    parser.add_argument('--max_length', type=int, default=None, help="maximum length of input tokens")
    parser.add_argument('--init_cross_attention_weights', action="store_true", default=False, help="whether to initialize second attention block with self attention weights")
    parser.add_argument('--max_context_length_per_k', type=int, default=128, help="maximum key/value pair length per k. Reduce this number to lower memory usage but the context will be truncated")
    parser.add_argument('--data_folder', default="processed_data", help="folder with your dataset files")
    parser.add_argument('--save_head', action="store_true", default=False, help="whether to save the LM head when saving the model")
    parser.add_argument('--data_config', help="A data_config.json file", default=None, help="path to the data config relative to data_folder")
    parser.add_argument('--output', default="retrieval_trained", help="output folder path")
    args = parser.parse_args()
    torch.manual_seed(0)

    args.device = torch.device("cuda") if not args.no_gpu else torch.device("cpu")
    logging.info("device available" + str(args.device))
    args.batch_size = args.batch_size_per_gpu * torch.cuda.device_count() if not args.no_gpu else args.batch_size_per_gpu
    logging.info("total batch size")
    args.config = AutoConfig.from_pretrained(args.model_path)
    model_type = args.config.model_type
    if not args.from_checkpoint:
        logging.info("training from Huggingface pretrained model. updating config.")
        if not args.not_retrieval:
            update_config_retrieval(args)
            logging.info("SUREALM config updated")
        else:
            update_config_regular(args)
            logging.info("baseline config updated")
    else:
        args.has_cross_attention = not args.not_cross_attention

    args.device = torch.device("cuda") if not args.no_gpu else torch.device("cpu")
    logging.info("Creating output folder: "+args.output)
    if os.path.exists(args.output):
        if not args.eval_only:
            raise RuntimeError("Output folder already exists.")

    # Write train script to output path
    os.makedirs(args.output, exist_ok=True)

    if args.data_config is None:
        logging.info("data config not set, defaulting to data_config.json")
        args.data_config = os.path.join(args.data_folder, "data_config.json")
    else:
        logging.info("loading data config "+ os.path.join(args.data_folder, args.data_config))
        args.data_config = os.path.join(args.data_folder, args.data_config)
    data_config_path = os.path.join(args.output, 'data_config.json')
    logging.info("copying data config and train script to output directory")
    copyfile(args.data_config, data_config_path)

    train_script_path = os.path.join(args.output, 'train_script.py')
    copyfile(__file__, train_script_path)
    with open(train_script_path, 'a') as fOut:
        fOut.write("\n\n# Script was called via:\n#python " + " ".join(sys.argv))

    #Load data config
    logging.info("loading data config")
    with open(args.data_config) as fIn:
        data_config = json.load(fIn)
    args.files = []

    if not args.eval_only:
        logging.info("current task is training")
        args.total_lines = 0
        for data in data_config:
            args.files.append(data)
            args.total_lines += data["lines"]
        if args.steps is None:
            args.steps = math.ceil(args.total_lines / args.batch_size)
        args.total_steps = args.steps * args.epochs
        logging.info("Total number of files: " + str(len(args.files)))
        logging.info("Total number of epochs: " + str(args.epochs))
        logging.info("Steps per epochs: " + str(args.steps))
        logging.info("Total steps: " + str(args.total_steps))

    else:
        logging.info("current task is evaluation")
        for data in data_config:
            args.files.append(data)
            logging.info("Total number of files: " + str(len(args.files)))

    logging.info("loading tokenizer")
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if args.tokenizer.pad_token is None:
        logging.info("pad token is not set, defaulting to eos")
        args.tokenizer.pad_token = args.tokenizer.eos_token
    is_baseline = not args.from_checkpoint and args.not_retrieval
    if model_type == "bert":  # Currently supporting three architectures
        model_class = RetrievalGenerationModel
    elif model_type == "roberta":
        model_class = RetrievalGenerationModelRoberta
    elif model_type == "gpt2":
        model_class = RetrievalGenerationModelGPT2

    if args.from_scratch:
        logging.info("training from scratch")
        try:
            args.model = model_class(config=args.config, args=args)
        except:
            args.model = model_class.from_config(config=args.config)
    else:
        logging.info("fine-tuning on pretrained model")
        try:
            args.model = model_class.from_pretrained(args.model_path, config=args.config, args=args)
        except:
            args.model = model_class.from_pretrained(args.model_path, config=args.config)
    if not is_baseline:
        logging.info("running SUREALM post post initialization")
        args.model.post_post_init(args)
    if torch.cuda.device_count() > 1 and not args.no_gpu:
        args.model = DataParallel(args.model, device_ids=list(range(torch.cuda.device_count())))
    
    ### Train Loop
    args.model = args.model.to(args.device)

    if not args.eval_only:
        args.optimizer = AdamW(params=args.model.parameters(), lr=args.lr, correct_bias=True)
        args.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=args.optimizer,
            num_warmup_steps=500,
            num_training_steps=args.total_steps,
        )
    handler = Handler(args=args)
    if args.eval_only:
        logging.info("start main evaluation loop")
        handler.evaluate()
        logging.info("evaluation done")
    else:
        logging.info("start main training loop")
        handler.train()
        logging.info("Training done")
    exit()
