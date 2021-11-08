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
from model.da_faster_rcnn.resnet import resnet
from model.da_faster_rcnn.vgg16 import vgg16

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import load_net, save_net, vis_detections
from roi_da_data_layer.roibatchLoader import roibatchLoader
from roi_da_data_layer.roidb import combined_roidb
from torch.autograd import Variable


import active_tools.chooseStrategy as CS
# from domain_tools.domain_classifier_util import Domain_classifier

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


import lib.model_tools.model_resource as MR
from lib.active_tools.strategy_enum import Strategy
from lib.active_tools.chooseStrategy import uncertain_sample


def get_random_list(train_txt_path):
    l=[]
    with open(train_txt_path,'r') as f:
        for line in f.readlines():
            line=line.strip('\n')
            l.append(line)
    import random
    random.shuffle(l)
    return l


def get_lc_list(dataset_name,net,imdb,roidb,ratio_list,ratio_index,class_agnostic,lc,gc,cuda_flag):
    model_dir=os.listdir(os.path.join("./data/experiments/SW_Faster_ICR_CCR",dataset_name,"model"))
    current_model,model_epoch=MR.get_current_model(model_dir)
    print(current_model)

    # initilize the network here.
    if net == "vgg16":
        fasterRCNN = vgg16(
            imdb.classes,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=class_agnostic,
            lc=lc,
            gc=gc,
        )
    elif net == "res101":
        fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=False,
            pretrained_path=None,
            class_agnostic=class_agnostic,
            lc=lc,
            gc=gc,
        )
    fasterRCNN.create_architecture()


    print("load checkpoint %s" % (current_model))
    checkpoint = torch.load(current_model)
    fasterRCNN.load_state_dict(
        {k: v for k, v in checkpoint["model"].items() if k in fasterRCNN.state_dict()}
    )

    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]
        print("load model successfully!")

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_cls_lb = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if cuda_flag:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        cfg.CUDA = True
        fasterRCNN.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_cls_lb = Variable(im_cls_lb)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    num_images = len(imdb.image_index)

    thresh = 0.0
    max_per_image = 100

    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    dataset = roibatchLoader(
        roidb,
        ratio_list,
        ratio_index,
        1,
        imdb.num_classes,
        training=False,
        normalize=False,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    data_iter = iter(dataloader)

    _t = {"im_detect": time.time(), "misc": time.time()}

    # 获取检测 bbox
    print("Evaluating detections")

    fasterRCNN.eval()
    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
    for i in range(num_images):

        data = next(data_iter)
        im_data.data.resize_(data[0].size()).copy_(data[0])
        im_info.data.resize_(data[1].size()).copy_(data[1])
        im_cls_lb.data.resize_(data[2].size()).copy_(data[2])
        gt_boxes.data.resize_(data[3].size()).copy_(data[3])
        num_boxes.data.resize_(data[4].size()).copy_(data[4])

        det_tic = time.time()

        with torch.no_grad():
            (
                rois,
                cls_prob,
                bbox_pred,
                category_cls_loss,
                rpn_loss_cls,
                rpn_loss_box,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                d_pixel,
                domain_p,
            ) = fasterRCNN(im_data, im_info, im_cls_lb, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if class_agnostic:
                    box_deltas = (
                            box_deltas.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = (
                            box_deltas.view(-1, 4)
                            * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda()
                            + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    )
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        # if vis:
        #     im = cv2.imread(imdb.image_path_at(i))
        #     im2show = np.copy(im)
        for j in xrange(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                # if vis:
                #     im2show = vis_detections(
                #         im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3
                #     )
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack(
                [all_boxes[j][i][:, -1] for j in xrange(1, imdb.num_classes)]
            )
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write(
            "im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r".format(
                i + 1, num_images, detect_time, nms_time
            )
        )
        sys.stdout.flush()

    detection_results=imdb.get_lc_sorted_list(all_boxes)

    sorted_list=uncertain_sample(detection_results,len(detection_results))

    for i in range(len(sorted_list)):
        sorted_list[i] = (sorted_list[i].split("/"))[-1]
        sorted_list[i]=sorted_list[i].split(".")[0]

    return sorted_list




def do_transfer(ratio,s_t_ratio,dataset_name,gpu_id,select_strategy,source_list,target_list,cuda_flag,net,lc,gc,class_agnostic):
    if torch.cuda.is_available() and not cuda_flag:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    np.random.seed(cfg.RNG_SEED)

    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_id)

    if dataset_name == "cityscapefoggy":
        print("loading our dataset...........")
        args_t_imdb_name = "cityscapefoggy_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]
    elif dataset_name == "clipart":
        args_t_imdb_name = "clipart_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]

    elif dataset_name == "watercolor":
        args_t_imdb_name = "watercolor_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "20",
        ]
    if dataset_name == "sim10k":
        print("loading our dataset...........")
        args_t_imdb_name = "sim10k_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]
    elif dataset_name=="cityscape10k":
        args_t_imdb_name = "cityscape10k_trainval"
        args_set_cfgs = [
            "ANCHOR_SCALES",
            "[8,16,32]",
            "ANCHOR_RATIOS",
            "[0.5,1,2]",
            "MAX_NUM_GT_BOXES",
            "30",
        ]
    args_cfg_file = (
        "cfgs/{}.yml".format(net)
    )

    if args_cfg_file is not None:
        cfg_from_file(args_cfg_file)
    if args_set_cfgs is not None:
        cfg_from_list(args_set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb, ratio_list, ratio_index = combined_roidb(args_t_imdb_name, False)


    current_model,model_epoch=MR.get_current_model(os.path.join("./data/experiments/SW_Faster_ICR_CCR",dataset_name,"model"))
    sorted_transfer_list=[]
    sorted_remove_list=[]

    if select_strategy==Strategy.random_strategy.value: #random
        print("random 迁移")
        sorted_transfer_list=get_random_list(os.path.join(imdb.get_dataset_path(),"ImageSets","Main","train.txt"))
    elif select_strategy==Strategy.lc_strategy.value:
        #加载最新模型计算不确定度
        print("lc 迁移")
        sorted_transfer_list=get_lc_list(dataset_name,net,imdb,roidb,ratio_list,ratio_index,class_agnostic,lc,gc,cuda_flag)
    elif select_strategy==Strategy.dt_t_strategy.value:
        print("域分类器 迁移目标域 dc-t")
        sorted_transfer_list=target_list
    elif select_strategy==Strategy.dt_t_lc_strategy.value:
        print("域分类器dc-t +lc ")
        l1=get_lc_list(dataset_name,net,imdb,roidb,ratio_list,ratio_index,class_agnostic,lc,gc,cuda_flag)
        l2=target_list
        sorted_dic = {}
        for i in range(0,min(len(target_list),len(l1))):
            item_1 = l2[i]
            item_2 = l1[i]
            if not sorted_dic.__contains__(item_1):
                sorted_dic[item_1] = 0
            if not sorted_dic.__contains__(item_2):
                sorted_dic[item_2] = 0

            sorted_dic[item_1] += i
            sorted_dic[item_2] += i

        sorted_transfer_list_tuple = sorted(sorted_dic.items(), key=lambda d: d[1], reverse=False)
        for item in sorted_transfer_list_tuple:
            sorted_transfer_list.append(item[0])

    elif select_strategy==Strategy.dt_t_s_strategy.value:
        print("域分类器 移除目标域并且迁移源域 dc-t-s")
        sorted_transfer_list=target_list
        sorted_remove_list=source_list
    else:
        raise NameError

    if len(sorted_remove_list)>0:
        imdb.remove_datas_from_source(sorted_remove_list)

    if len(sorted_transfer_list)>0:
        imdb.add_datas_from_target(sorted_transfer_list,float(ratio),model_epoch,s_t_ratio)













