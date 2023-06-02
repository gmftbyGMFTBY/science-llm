#!/bin/bash

deepspeed --include localhost:1,2 --master_addr 127.0.0.1 --master_port 28457 train_pretrain.py \
    --model scillm\
    --model_path decapoda-research/llama-7b-hf\
    --train_data_path ../data/redpajama_test_tokens.json \
    --test_data_path ../data/redpajama_test_tokens.json \
    --save_path ./ckpt/scillm/ \
    --log_path ./rest/
