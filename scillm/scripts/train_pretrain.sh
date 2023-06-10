#!/bin/bash

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_addr 127.0.0.1 --master_port 28457 train_pretrain.py \
    --model scillm\
    --model_path decapoda-research/llama-7b-hf\
    --train_data_path ../data/pretrain/train \
    --test_data_path ../data/pretrain/test/redpajama_tokens_test_v1.json \
    --save_path ./ckpt/scillm/ \
    --log_path ./rest/
