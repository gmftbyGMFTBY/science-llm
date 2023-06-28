#!/bin/bash

# for num in {0..15}
# do
#     CUDA_VISIBLE_DEVICES=3 python test_qasper.py \
#         --model scillm\
#         --model_path decapoda-research/llama-7b-hf\
#         --delta_model_path ../ckpt/scillm/$num\
#         --data_path ../data/sft/qasper/qasper_yes_no_test_sft.json
# done
    
# --model_path decapoda-research/llama-7b-hf\
# --model_path baichuan-inc/baichuan-7B\


# CUDA_VISIBLE_DEVICES=7 python test_qasper.py \
#     --model scillm\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path ../ckpt/scillm_baichuan/1\
#     --data_path ../data/sft/qasper/qasper_yes_no_test_sft.json
# exit

for num in {0..5}
do
    CUDA_VISIBLE_DEVICES=6 python test_qasper.py \
        --model scillm\
        --model_path baichuan-inc/baichuan-7B\
        --delta_model_path ../ckpt/scillm_baichuan/$num\
        --data_path ../data/sft/qasper/qasper_yes_no_test_sft.json
done
 
