#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test_ppl.py \
    --model scillm\
    --model_path decapoda-research/llama-7b-hf\
    --delta_model_path ckpt/scillm_backup/peft_model\
    --data_path ../data/pretrain/test/redpajama_tokens_test_v1.json
