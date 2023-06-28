#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python deploy.py \
#     --model scillm-sft\
#     --model_path baichuan-inc/baichuan-7B\
#     --delta_model_path ../ckpt/scillm-emotional-sft/18\
#     --port 23333


CUDA_VISIBLE_DEVICES=1 python deploy.py \
    --model scillm-sft\
    --model_path baichuan-inc/baichuan-7B\
    --delta_model_path ../ckpt/scillm-delta-18-baichuan-no-dev-with-scimrc-dolly-sft/18\
    --port 23334
