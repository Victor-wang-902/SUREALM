import argparse
import pickle
import json
from shutil import copyfile
import torch
import faiss
import transformers
from transformers import AutoTokenizer
import logging

from .preprocessing import (
    init_model, 
    new_faiss_index, 
    index_to_cpu, 
    index_to_gpu, 
    prepare_index, 
    skip_encode, 
    convert_embs, 
    prepare_fast_data
)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="encoder")
    parser.add_argument("--model_name", type=str, default="multi-qa-MiniLM-L6-cos-v1", required=False, help="sentence transformer for encoding")
    parser.add_argument("--target_tokenizer", type=str, default=None, help="the tokenizer for LM to train with the data prepared")
    parser.add_argument("--data_folder", type=str, required=True, help="raw data directory")
    parser.add_argument("--data_config", type=str, default="data_config.json", required=False, help="name of the data config file WITHIN data_folder")
    parser.add_argument("--not_preserve_position_embedding", action="store_true", default=False, help="set to true if sentence transformer does not accept explicit positional embedding input")
    parser.add_argument("--out_folder", type=str, default="processed_data", required=False, help="output directory")
    parser.add_argument("--cpu", action="store_true", help="whether to only use CPUs")
    args = parser.parse_args()
    args.device = torch.device('cuda') if torch.cuda.is_available() and not args.cpu else torch.device('cpu')
    if args.target_tokenizer is not None:
        args.target_tokenizer = AutoTokenizer.from_pretrained(args.target_tokenizer)
    logging.info("device available:" + str(args.device))
    args.model = init_model(args)
    args.model.to(args.device)
    args.model.eval()
    emb_dim  = args.model.config.hidden_size
    logging.info("hidden size of embeddings" + str(emb_dim))
    args.raw_data_config = os.path.join(args.data_folder, args.data_config)
    with open(args.raw_data_config, "r") as f:
        data_config = json.load(f)
    args.files = [data for data in data_config]
    args.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
    if not os.path.exists(args.out_folder):
        logging.info("creating directory: " + args.out_folder)
        os.makedirs(args.out_folder)
    for file in args.files:
        logging.info("start processing dataset: " + file["name"])
        folder_name = os.path.splitext(file["name"])[0] + "_" + str(file["ngram"]) + "gram" + "_" + "top" + str(file["topk"]) + "_" + "win" + str(file["window_size"])
        logging.info("processed data will be saved in " + os.path.join(args.out_folder, folder_name))
        if os.path.exists(os.path.join(args.out_folder, folder_name)):
            logging.info("folder already exists, skipping" + folder_name)
            continue
        args.pre_index = new_faiss_index(emb_dim, args)
        if args.device != torch.device("cpu"):
            args.pre_index = index_to_gpu(args.pre_index)
        logging.info("encoding raw data")
        # preprocess the data through encoding with sentence transformers
        emb_map, str_map, pre_embs, \
        post_embs, val_emb_map, val_pre_embs, \
        test_emb_map, test_pre_embs = prepare_index(file, args)
        pre_embs, post_embs = convert_embs(args, pre_embs, post_embs)
        _, val_pre_embs = convert_embs(args, post_embs=val_pre_embs)
        if test_emb_map is not None:
            _, test_pre_embs = convert_embs(args, post_embs=test_pre_embs)
        original_name = file["name"]
        file["name"] = folder_name
        os.makedirs(os.path.join(args.out_folder, folder_name))
        if args.device != torch.device("cpu"):
            args.pre_index = index_to_cpu(args.pre_index)
        # saving intermediate processed data.
        logging.info("done. Writing processed data.")
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
        with open(os.path.join(args.out_folder, folder_name, "string_mappings.pkl"), "wb") as f:
            pickle.dump(str_map, f)
        with open(os.path.join(args.out_folder, folder_name, "val_embedding_mappings.pkl"), "wb") as f:
            pickle.dump(val_emb_map, f)
        copyfile(os.path.join(args.data_folder, original_name, "train.txt"), os.path.join(args.out_folder, folder_name, "data.txt"))
        if os.path.exists(os.path.join(args.data_folder, original_name, "test.txt")):
            copyfile(os.path.join(args.data_folder, original_name, "test.txt"), os.path.join(args.out_folder, folder_name, "data_test.txt"))
        copyfile(os.path.join(args.data_folder, original_name, "val.txt"), os.path.join(args.out_folder, folder_name, "data_val.txt"))
        logging.info("total vectors indexed for " + file["name"] +": " + str(args.pre_index.ntotal))
        # precompute retrieved embeddings for fast training.
        logging.info("precomputing processed data embedding retrieval")
        fast_idx, fast_pre_embs, fast_post_embs, \
        val_fast_idx, val_fast_pre_embs, val_fast_post_embs, \
        test_fast_idx, test_fast_pre_embs, test_fast_post_embs = prepare_fast_data(
            args, file, emb_map, 
            pre_embs, post_embs, val_emb_map, 
            val_pre_embs, test_emb_map, test_pre_embs
        )
        logging.info("saving final preprocessed data ready for training.")
        # saving final processed data.
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
        if os.path.exists(os.path.join(args.out_folder, "data_config.json")):
            with open(os.path.join(args.out_folder, "data_config.json"), "r") as f:
                existing_json = json.load(f)
        else:
            existing_json = []
        existing_json.extend(args.files)
        args.files = existing_json
        logging.info("updating configs")
        with open(os.path.join(args.out_folder, "data_config.json"), "w") as f:
            json.dump(args.files, f)
        logging.info("preprocessing done")


if __name__ == "__main__":
    main()
