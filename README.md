# SUffix Retirieval-Augmented Language Modeling
This repository contains tha data, scripts and source codes for our paper submission at ICASSP 2023.

We share our source code for indexing, precomputing retrieval information, training, evaluation, as well as online generation. We modified code from Huggingface and introduced wrapper model classes for implementation of our model such that we support model initialization from multiple Huggingface Transformer model types.

In our current implementation, we only present code for training with offline pre-computation of retrieval embeddings as this is the most efficient way to train our model.
