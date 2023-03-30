#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../


master_port=12563

CUDA_VISIBLE_DEVICES=2,3 \
python -m torch.distributed.launch --nproc_per_node 2  \
    --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=${master_port} \
    uniform_finetune.py   --model_type chatglm --model_name_or_path THUDM/chatglm-6b \
    --data alpaca-belle-cot --lora_target_modules query_key_value \
    --lora_r 32 --lora_alpha 32 --lora_dropout 0.1 --per_gpu_train_batch_size 8 \
    --learning_rate 2e-5 --epochs 5