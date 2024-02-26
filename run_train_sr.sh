#!/bin/bash
random_port=$((1024 + RANDOM % 64512))
num_gpus=2
node_num=1
export MASTER_PORT=$random_port
#export NCLL_P2P_LEVEL=NVL

srun \
-p ai4earth \
-J DARS \
--kill-on-bad-exit=1 \
--quotatype=reserved \
--gres=gpu:$num_gpus \
accelerate launch --main_process_port=$random_port --num_processes=$num_gpus main_downscale.py \
--config ./configs/train_downscale_sr.yaml --grid
#-o fixed_log/%j.out \