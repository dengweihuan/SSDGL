#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`
config_path='SSDGL.SSDGL_1_0_Indianpine_small'

model_dir='./log/Indianpine_small/SSDGL/1.0_poly'


python train.py \
    --config_path=${config_path} \
    --model_dir=${model_dir} \
    train.save_ckpt_interval_epoch 9999
