#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python test_generation.py \
    --model scillm-sft\
    --model_path baichuan-inc/baichuan-7B\
    --delta_model_path ../ckpt/scillm-baichuan-sft/18\
    --data_path ../data/sft/qasper/qasper_test_sft.json\
    --result_path ../rest/ours_baichuan_qasper_test_sft.txt
    
# --model_path decapoda-research/llama-7b-hf\
# --model_path baichuan-inc/baichuan-7B\
