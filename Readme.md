# SUffix Retirieval-Augmented Language Modeling
This repository contains tha data, scripts and source codes for our paper submission at ICASSP 2023.

We share our source code for indexing, precomputing retrieval information, training, evaluation, as well as online generation. We modified code from Huggingface and introduced wrapper model classes for implementation of our model such that we support model initialization from multiple Huggingface Transformer model types.

In our current implementation, we only present code for training with offline pre-computation of retrieval embeddings as this is the most efficient way to train our model.

# Dependencies
```
pip install -r requirements.txt
```

# Indexing
Here we show an example of indexing a faiss knowledge base from our data folder and precompute prefix-suffix embedding pairs using [sentence-transformers/multi-qa-distilbert-cos-v1](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1) for [bert-base-uncased](https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.).
```
./encode.sh
```
Alternatively, if the SUREALM model tokenizer, say BPE for GPT-2, is different from that of the sentence transformer used during indexing, one needs to set `target_tokenizer`. For example:
```
./encode_gpt2.sh
```

# Training
Suppose one did indexing by running `./encode.sh`, they can then train a [bert-base-uncased](https://huggingface.co/bert-base-uncased?text=The+goal+of+life+is+%5BMASK%5D.) model with:
```
./train.sh
```
Training other model architectures is as trivial as changing the `model_path` argument, as long as the tokenizer during indexing and training is compatible.

# Inference (Generation)
After obtaining a desired model checkpoint, one can run SUREALM generation interactively with retrieval by:
```
./inference.sh
```
Alternatively, one can delete the `--interactive` argument and provide `--prompt_file`:
```
./inference_prompt.sh
```
If SUREALM has a different tokenizer than the desired sentence transformer, one needs to provide `--tokenizer_difference` argument. For example, if one wants to train [GPT-2](https://huggingface.co/gpt2?text=My+name+is+Mariama%2C+my+favorite) with [`multi-qa-distilbert-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1):
```
./inference_gpt2.sh
```
