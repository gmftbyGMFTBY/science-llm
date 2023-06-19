#!/bin/bash

deepspeed --include localhost:5 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model scillm-sft\
    --model_path baichuan-inc/baichuan-7B\
    --delta_model_path None\
    --train_data_path ../data/sft/emotional/train.json\
    --save_path ./ckpt/scillm-emotional-sft/ \
    --log_path ./rest/scillm-emotional-sft
    
# --model_path decapoda-research/llama-7b-hf\
