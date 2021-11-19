#!/bin/bash
dataset="watercolor"
save_dir="./data/experiments/SW_Faster_ICR_CCR/watercolor/model"
pretrained_path="./data/pretrained_model/vgg16_caffe.pth"
net="vgg16"
#checkpoint_interval=4
GPUID=0

round_num=20

st_ratio=10 #enhance

lr=0.001

max_transfer_num=117
# st_ratio=1 #not enhance

#0 random
#1 lc
#2 dc-t
#3 dc-t +lc

select_strategy =0
#at 0
python da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --lr ${lr} --da_use_contex --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio} --select_strategy ${select_strategy} --max_transfer_num ${max_transfer_num}
