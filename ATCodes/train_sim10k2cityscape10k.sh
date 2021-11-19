#!/bin/bash
dataset="cityscape10k"
save_dir="./data/experiments/SW_Faster_ICR_CCR/cityscape10k/model"
pretrained_path="./data/pretrained_model/vgg16_caffe.pth"
net="vgg16"
#checkpoint_interval=4
GPUID=0

round_num=20

st_ratio=10 #enhance

lr=0.0001

max_transfer_num=99
# st_ratio=1 #not enhance


#da_train_swiicrccr 直接先域适应
#python da_train_swicrccr.py --max_epochs 14 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --checkpoint_interval ${checkpoint_interval} --gc --lc --da_use_contex --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio}

#at
#python da_tr
# st_ratio=1 #not enhanceain_net2.py --max_epochs 14 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --checkpoint_interval ${checkpoint_interval} --gc --lc --da_use_contex --r true --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio}


#0 random
#1 lc
#2 dc-t
#3 dc-t +lc

select_strategy =0
#at 0
python da_train_net.py --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --lr ${lr} --da_use_contex --round_num ${round_num} --gpu_id ${GPUID} --st_ratio ${st_ratio} --select_strategy ${select_strategy} --max_transfer_num ${max_transfer_num}
