#!/bin/bash


# CUDA_VISIBLE_DEVICES=3 python test_ppl.py \
#     --model scillm\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path ../ckpt/scillm/$num\
#     --data_path ../data/pretrain/test/redpajama_tokens_test_v1_chinese.json
# exit

for num in {0..5}
do
    CUDA_VISIBLE_DEVICES=6 python test_ppl.py \
        --model scillm\
        --model_path baichuan-inc/baichuan-7B\
        --delta_model_path ../ckpt/scillm_baichuan/$num\
        --data_path ../data/pretrain/test/redpajama_tokens_test_v1_chinese.json
done
    
# --model_path decapoda-research/llama-7b-hf\
# --model_path baichuan-inc/baichuan-7B\
# --data_path ../data/pretrain/test/redpajama_tokens_test_v1_chinese.json
# --data_path ../data/pretrain/test/redpajama_tokens_test_v1.json
