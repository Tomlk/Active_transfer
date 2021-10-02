#!/bin/bash

dataset="watercolor"
save_dir="./data/experiments/SW_Faster_ICR_CCR/watercolor/model"
pretrained_path="./data/pretrained_model/vgg16_caffe.pth"
net="vgg16"
checkpoint_interval=4
GPUID=0

round_num=20

st_ratio=22


#da_train_swiicrccr 直接先域适应
# python da_train_swicrccr.py --max_epochs 14 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --checkpoint_interval ${checkpoint_interval} --gc --lc --da_use_contex --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio}


python da_train_net2.py --max_epochs 14 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --checkpoint_interval ${checkpoint_interval} --gc --lc --da_use_contex --r true --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio}