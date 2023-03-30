#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


master_port=12363

CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.launch --nproc_per_node 2  \
    --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=${master_port} \
    uniform_finetune.py   --model_type bloom --model_name_or_path bigscience/bloomz-7b1-mt \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --per_gpu_train_batch_size 16 --learning_rate 3e-4 --epochs 5