#!/bin/bash

. ./path.sh || exit 1;

conda activate m2s_avsr


dir=exp

average_num=10


### for step average
average_mode=step
max_step=88888888
echo "Average models by step..."
decode_checkpoint=$dir/avg${average_num}_${average_mode}.pt
echo "do model average and final checkpoint is $decode_checkpoint"
python wenet/bin/average_model.py \
    --dst_model $decode_checkpoint \
    --src_path $dir  \
    --num ${average_num} \
    --mode ${average_mode} \
    --max_step ${max_step} \
    --val_best


