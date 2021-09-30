
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



args_dataset = "cityscape"
args_num_epoch = -1
args_output_dir = "./"
args_cfg = "cfgs/vgg16.yml"
args_net = "vgg16"
args_model_dir = "models.pth"
args_part = "test_t"
args_cuda = True
args_large_scale = False
args_class_agnostic = False
args_ls = True
args_mGPUs = False
args_parallel_type = 0
args_checksession = 1
args_checkepoch = 1
args_checkpoint = 10021
args_model_name = ""
args_USE_cls_cotrain = True
args_USE_box_cotrain = True
args_lc = True
args_gc = True
args_ratio = 0.05
args_epoch_index = 12
args_st_ratio = 1
args_test_flag = False

args_s_imdb_name = ""
args_s_imdbtest_name = ""
args_t_imdb_name = ""
args_t_imdbtest_name = ""
args_set_cfgs = []
args_vis = False

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def get_test_boxes(imdb, roidb, ratio_list, ratio_index,faster_rcnn):
    max_per_image=100

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    im_cls_lb = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args_cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        im_cls_lb = im_cls_lb.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    im_cls_lb = Variable(im_cls_lb)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args_cuda:
        cfg.CUDA = True

    if args_cuda:
        faster_rcnn.cuda()
    vis = args_vis

    if vis:
        thresh = 0.05
    else:
        thresh = 0.0

    save_name = args_part + args_model_dir.split("/")[-1]
    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in xrange(num_images)] for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, save_name)
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
    det_file = os.path.join(output_dir, "detections.pkl")

    faster_rcnn.eval()
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
            ) = faster_rcnn(im_data, im_info, im_cls_lb, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                # Optionally normalize targets by a precomputed mean and stdev
                if args_class_agnostic:
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
                if args_class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4: (j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                # keep = nms(cls_dets, cfg.TEST.NMS)
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if vis:
                    im2show = vis_detections(
                        im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3
                    )
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

        # if vis:
        #     cv2.imwrite("result.png", im2show)
        #     pdb.set_trace()
        #     # cv2.imshow('test', im2show)
        #     # cv2.waitKey(0)

    return all_boxes


def execute_transfer_data(epoch_index, ratio, s_t_ratio,dataset,target_list,source_list,lc_flag,random_flag=0):

    np.random.seed(cfg.RNG_SEED)

    from da_test_net import get_last_model_path
    args_model_dir = get_last_model_path(dataset)

    if dataset == "cityscapefoggy":
        args_t_imdb_name = "cityscapefoggy_trainval"
    elif dataset == "clipart":
        args_t_imdb_name = "clipart_trainval"
    elif dataset == "watercolor":
        print("loading our dataset...........")
        args_t_imdb_name = "watercolor_trainval"

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
    all_boxes=get_test_boxes(imdb, roidb, ratio_list, ratio_index,fasterRCNN)

    #对目标域 train 检测不确定度
    detection_for_all_images = imdb.evaluate_detections(all_boxes, None, None,True)
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
        if args_test_flag == False:  # TODO
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
        if args_test_flag == False:  # TODO
            print("开始迁移目标域数据...")
            imdb.add_datas_from_target(_target_list, float(args_ratio), args_epoch_index, args_st_ratio)


    if random_flag==1:
        import lib.active_tools.chooseStrategy as CS
        random_list = CS.random_sample(os.path.join(imdb.get_dataset_path(),"JPEGImages"))
        l=[]
        for item in random_list:
            l.append(item.split('/')[-1])
        imdb.add_datas_from_target(l, float(args_ratio), args_epoch_index, args_st_ratio)

