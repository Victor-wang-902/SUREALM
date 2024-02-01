#!/bin/bash

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ext3/miniconda3/envs/rblm/lib

##python encoder.py --data_path test.txt --str_map --out_folder new_indices_with_mappings --cpu
python search_example.py