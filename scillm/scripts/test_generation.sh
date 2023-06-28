#!/bin/bash

# CUDA_VISIBLE_DEVICES=6 python test_generation.py \
#     --model scillm-sft\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path ../ckpt/scillm-scratch-baichuan-no-dev-with-scimrc-sft/18\
#     --data_path ../data/sft/processed_qasper_test_set.json\
#     --result_path ../rest/baichuan_gt.txt\
#     --recall False
# exit

CUDA_VISIBLE_DEVICES=7 python test_generation.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ../ckpt/scillm-delta-18-llama-no-dev-with-scimrc-sft-new-v2/18\
    --data_path ../data/sft/processed_qasper_test_set.json\
    --result_path ../rest/llama_qasper_recall_test.txt\
    --recall True
exit
CUDA_VISIBLE_DEVICES=6 python test_generation.py \
    --model scillm-sft\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ../ckpt/scillm-delta-18-llama-no-dev-with-scimrc-sft-new/18\
    --data_path ../data/sft/processed_qasper_test_set.json\
    --result_path ../rest/llama_qasper_gt.txt\
    --recall False &
 
    
# --delta_model_path ../ckpt/scillm-delta-18-llama-no-dev-with-scimrc-sft-new/18\
# --model_path decapoda-research/llama-7b-hf\
# --model_path baichuan-inc/baichuan-7B\
# --delta_model_path ../ckpt/scillm-delta-16-llama-no-dev-with-scimrc-sft/18\
