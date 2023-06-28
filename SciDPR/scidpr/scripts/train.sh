#!/bin/bash

export NCCL_IB_DISABLE=1

# ========== metadata ========== #
dataset=$1
model=$2
cuda=$3 
# ========== metadata ========== #

root_dir=$(cat config/base.yaml | shyaml get-value root_dir)
version=$(cat config/base.yaml | shyaml get-value version)

# backup
recoder_file=$root_dir/rest/$dataset/$model/recoder_$version.txt

echo "find root_dir: $root_dir"
echo "find version: $version"
echo "write running log into recoder file: $recoder_file"


gpu_ids=(${cuda//,/ })
CUDA_VISIBLE_DEVICES=$cuda python -m torch.distributed.launch --nproc_per_node=${#gpu_ids[@]} --master_addr 127.0.0.1 --master_port 28458 train.py \
    --dataset $dataset \
    --model $model \
    --multi_gpu $cuda \
    --total_workers ${#gpu_ids[@]}
