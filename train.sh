#!/bin/bash
config_file=./config/EMFormer.yaml
config='EMFormer'
run_num='1'

NAME="EMFormer_train"

checkpoint=""

LOG_DIR="./logs/${NAME}/"
mkdir -p -- "$LOG_DIR"

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m torch.distributed.run --nproc_per_node=4 train.py \
            --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num --exp_dir=$LOG_DIR --checkpoint=$checkpoint \
            > ${LOG_DIR}train.log 2>&1 &