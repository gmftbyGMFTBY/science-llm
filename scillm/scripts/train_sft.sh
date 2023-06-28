#!/bin/bash

# deepspeed --include localhost:4,5 --master_addr 127.0.0.1 --master_port 28457 train_sft.py \
#     --model scillm-sft\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path None\
#     --train_data_path ../data/sft/train.json\
#     --save_path ../ckpt/scillm-baichuan-sft/ \
#     --log_path ../rest/scillm-baichuan-sft
# exit
 

deepspeed --include localhost:3,4 --master_addr 127.0.0.1 --master_port 28456 train_sft.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ../ckpt/scillm/18\
    --train_data_path ../data/sft/train.json\
    --save_path ../ckpt/scillm-delta-18-llama-no-dev-with-scimrc-sft-new-v2/ \
    --log_path ../rest/scillm-delta-18-llama-no-dev-with-scimrc-sft-new-v2
    
# --model_path decapoda-research/llama-7b-hf\
