#!/bin/bash

# deepspeed --include localhost:4,5 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
#     --model scillm-sft\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path None\
#     --train_data_path ../data/sft/train.json\
#     --save_path ../ckpt/scillm-baichuan-sft/ \
#     --log_path ../rest/scillm-baichuan-sft
# exit
 

deepspeed --include localhost:3,4 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path None\
    --train_data_path ../data/sft/train.json\
    --save_path ../ckpt/scillm-llama-scratch-sft/ \
    --log_path ../rest/scillm-llama-scratch-sft
    
# --model_path decapoda-research/llama-7b-hf\
