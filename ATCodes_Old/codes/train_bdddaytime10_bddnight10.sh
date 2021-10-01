#!/bin/bash

dataset="bddnight10"
save_dir="./data/experiments/SW_Faster_ICR_CCR/bddnight10/model"
pretrained_path="./data/pretrained_model/vgg16_caffe.pth"
net="vgg16"
checkpoint_interval=4
GPUID=2

round_num=20

st_ratio=2 #enhance


python da_train_net.py --max_epochs 12 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --checkpoint_interval ${checkpoint_interval} --gc --lc --da_use_contex --r true --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio}