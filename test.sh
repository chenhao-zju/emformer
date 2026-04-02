config_file=./config/EMFormer.yaml    
config='EMFormer'                    
run_num='1'

NAME=""


LOG_DIR="./logs/${NAME}/test_global/"
mkdir -p -- "$LOG_DIR"

WEIGHTS="./logs/${NAME}/${config}/1/training_checkpoints/best_ckpt.tar"

CUDA_VISIBLE_DEVICES=2 nohup python test.py \
            --yaml_config=$config_file --config=$config --run_num=$run_num --override_dir=$LOG_DIR \
            --weights=$WEIGHTS > ${LOG_DIR}test_global.log 2>&1 &
            