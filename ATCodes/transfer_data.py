
from __future__ import absolute_import, division, print_function

import argparse
import os
import pdb
import pickle
import pprint
import sys
import time

import _init_paths
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# from model.faster_rcnn.vgg16 import vgg16
from lib.model.da_faster_rcnn.resnet import resnet
from lib.model.da_faster_rcnn.vgg16 import vgg16

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import load_net, save_net, vis_detections
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3

args_output_dir = "./"
args_net = "vgg16"
args_cuda = True
args_large_scale = False
args_class_agnostic = False
args_lc = True
args_gc = True
args_vis = False

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

from lib.detection_boxes_tools import detection_boxes

def execute_transfer_data(epoch_index, ratio, s_t_ratio,dataset,target_list,source_list,lc_flag,random_flag=0):

    np.random.seed(cfg.RNG_SEED)

    from da_test_net import get_last_model_path
    args_model_dir = get_last_model_path(dataset)

    if dataset == "cityscapefoggy":
        args_t_imdb_name = "cityscapefoggy_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]

    elif dataset == "clipart":
        args_t_imdb_name = "clipart_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    elif dataset == "watercolor":
        print("loading our dataset...........")
        args_t_imdb_name = "watercolor_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    args_cfg_file = (
        "cfgs/{}_ls.yml".format(args_net)
        if args_large_scale
        else "cfgs/{}.yml".format(args_net)
    )

    if args_cfg_file is not None:
        cfg_from_file(args_cfg_file)
    if args_set_cfgs is not None:
        cfg_from_list(args_set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    #目标域训练集:train
    imdb, roidb, ratio_list, ratio_index = combined_roidb(
        args_t_imdb_name, False
    )
    imdb.competition_mode(on=True)
    print("{:d} roidb entries".format(len(roidb)))

    load_name = args_model_dir
    print(load_name)

    # initilize the network here.
    if args_net == "vgg16":
        fasterRCNN = vgg16(
            imdb.classes,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=args_class_agnostic,
            lc=args_lc,
            gc=args_gc,
        )
    elif args_net == "res101":
        fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=args_class_agnostic,
            lc=args_lc,
            gc=args_gc,
        )
    elif args_net == "res50":
        fasterRCNN = resnet(
            imdb.classes, 50, pretrained=False, class_agnostic=args_class_agnostic
        )
    elif args_net == "res152":
        fasterRCNN = resnet(
            imdb.classes, 152, pretrained=False, class_agnostic=args_class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(
        {k: v for k, v in checkpoint["model"].items() if k in fasterRCNN.state_dict()}
    )
    # fasterRCNN.load_state_dict(checkpoint['model'])
    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    print("load model successfully!")

    start = time.time()
    all_boxes=detection_boxes.get_test_boxes(imdb, roidb, ratio_list, ratio_index,fasterRCNN)

    #对目标域 train 检测不确定度
    detection_for_all_images = imdb.get_detection_boxes_result(all_boxes)
    end = time.time()
    print("不确定度 time: %0.4fs" % (end - start))

    # with open(det_file, "wb") as f:
    with open("predict_all_boxes.pkl", "wb") as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print("Evaluating detections")

    if not os.path.exists(args_output_dir):
        os.makedirs(args_output_dir)

    _source_list=source_list
    _target_list=target_list
    _lc_flag=lc_flag
    args_ratio=ratio
    args_st_ratio=s_t_ratio
    args_epoch_index=epoch_index

    if _source_list is not None:
        print("_source_list:")
        for i in range(len(_source_list)):
            _source_list[i] = (_source_list[i].split("/"))[-1]
        print(_source_list)

        # 如果是train数据，转移图像文件及标注文件
        print("开始移除源域数据....")
        imdb.remove_datas_from_source(_source_list, float(args_ratio), args_st_ratio)

    if _target_list is not None:
        print("_target_list:")
        for i in range(len(_target_list)):
            _target_list[i] = (_target_list[i].split("/"))[-1]
        print(_target_list)

        if _lc_flag==1:
            import lib.active_tools.chooseStrategy as CS
            uncertain_list = CS.uncertain_sample(detection_for_all_images, len(detection_for_all_images))
            for i in range(len(uncertain_list)):
                uncertain_list[i] = (uncertain_list[i].split("/"))[-1]

            # 与target_list 权重1：1处理排序。
            i = 0
            sorted_dic = {}
            for i in range(0,min(len(_target_list),len(uncertain_list))):
                item_1 = _target_list[i]
                item_2 = uncertain_list[i]
                if not sorted_dic.__contains__(item_1):
                    sorted_dic[item_1] = 0
                if not sorted_dic.__contains__(item_2):
                    sorted_dic[item_2] = 0

                sorted_dic[item_1] += i
                sorted_dic[item_2] += i

            sorted_target_list_tuple = sorted(sorted_dic.items(), key=lambda d: d[1], reverse=False)

            sorted_target_list = []
            for item in sorted_target_list_tuple:
                sorted_target_list.append(item[0])

            _target_list=sorted_target_list

        # 如果是train数据，转移图像文件及标注文件
        print("开始迁移目标域数据...")
        imdb.add_datas_from_target(_target_list, float(args_ratio), args_epoch_index, args_st_ratio)


    if random_flag==1:
        import lib.active_tools.chooseStrategy as CS
        random_list = CS.random_sample(os.path.join(imdb.get_dataset_path(),"JPEGImages"))
        l=[]
        for item in random_list:
            l.append(item.split('/')[-1])
        imdb.add_datas_from_target(l, float(args_ratio), args_epoch_index, args_st_ratio)


