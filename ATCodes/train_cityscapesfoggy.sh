#!/bin/bash
save_dir="./data/experiments/SW_Faster_ICR_CCR/cityscapesfoggy/model"
dataset="cityscapesfoggy"
pretrained_path="./data/pretrained_model/vgg16_caffe.pth"
net="vgg16"
gpu_id=1
python da_train_net.py --max_epochs 20 --cuda --dataset ${dataset} --net ${net} --save_dir ${save_dir} --pretrained_path ${pretrained_path} --gc --lc --da_use_contex  --gpu_id ${gpu_id}
