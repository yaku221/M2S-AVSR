#!/bin/bash

. ./path.sh || exit 1;

conda activate m2s_avsr

export CUDA_VISIBLE_DEVICES="0,1"
echo "CUDA_VISIBLE_DEVICES is ${CUDA_VISIBLE_DEVICES}"


HOST_NODE_ADDR="localhost:0"
num_nodes=1
job_id=train_m2s

data_type=raw

work_path=/path/to/m2s_avsr


data_path=${work_path}/data
## audio data
train_audio_set=${data_path}/audio/train
dev_audio_set=${data_path}/audio/dev


## video data
train_video_set=${data_path}/video/train
dev_video_set=${data_path}/video/dev


train_config=conf/m2s_avsr/m2s_avsr_whisper_large.yaml


checkpoint=

dir=exp

tensorboard_dir=tensorboard

num_workers=8
prefetch=10


# for training
train_engine=deepspeed
deepspeed_config=conf/whisper/ds_stage1.json
deepspeed_save_states="model+optimizer"


mkdir -p $dir
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
dist_backend="nccl"

if [ ${train_engine} == "deepspeed" ]; then
  echo "$0: using deepspeed"
else
  echo "$0: using torch ddp"
fi


echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
          --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
  wenet/bin/train_avsr.py \
    --train_engine ${train_engine} \
    --config $train_config \
    --data_type  $data_type \
    --train_audio_data ${train_audio_set}/data.list \
    --train_video_data ${train_video_set}/data.list \
    --cv_audio_data ${dev_audio_set}/data.list \
    --cv_video_data ${dev_video_set}/data.list \
    ${checkpoint:+--checkpoint $checkpoint} \
    --model_dir $dir \
    --tensorboard_dir ${tensorboard_dir} \
    --ddp.dist_backend $dist_backend \
    --num_workers ${num_workers} \
    --prefetch ${prefetch} \
    --pin_memory \
    --print_model \
    --deepspeed_config ${deepspeed_config} \
    --deepspeed.save_states ${deepspeed_save_states}







