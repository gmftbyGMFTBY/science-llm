#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_generation.py \
    --model scillm-sft\
    --model_path baichuan-inc/baichuan-7B\
    --delta_model_path ckpt/scillm-emotional-sft/8\
    --data_path ../data/sft/qasper/qasper_test_sft.json\
    --result_path ./rest/ours_qasper_test_sft.txt
    
# --model_path decapoda-research/llama-7b-hf\
