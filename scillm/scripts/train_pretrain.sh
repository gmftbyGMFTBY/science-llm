#!/bin/bash

deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28457 train_pretrain.py \
    --model scillm\
    --model_path /home/johnlan/pretrained_models/LLaMA-7B-HF/ \
    --train_data_path ../data/redpajama_test_tokens.json \
    --test_data_path ../data/redpajama_test_tokens.json \
    --save_path ./ckpt/scillm/ \
    --log_path ./rest/
