#!/bin/bash

CUDA_VISIBLE_DEVICES=5 python test_ppl.py \
    --model scillm\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ckpt/scillm/18\
    --data_path ../data/pretrain/test/redpajama_tokens_test_v1.json
    
# --model_path decapoda-research/llama-7b-hf\
# --model_path baichuan-inc/baichuan-7B\
# --data_path ../data/pretrain/test/redpajama_tokens_test_v1_chinese.json
