#!/bin/bash

CUDA_VISIBLE_DEVICES=4 python test_qasper.py \
    --model scillm\
    --model_path baichuan-inc/baichuan-7B\
    --delta_model_path ckpt/scillm_baichuan/3\
    --data_path ../data/sft/qasper/qasper_yes_no_test_sft.json
    
# --model_path decapoda-research/llama-7b-hf\
