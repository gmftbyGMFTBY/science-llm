#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python inference.py \
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ckpt/scillm/_0\
    --max_length 4096\
    --generate_len 512\
    --top_k 50\
    --top_p 0.92
