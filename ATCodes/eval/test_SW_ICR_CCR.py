# from __future__ import absolute_import, division, print_function
#
# import argparse
# import os
# import pdb
# import pickle
# import pprint
# import sys
# import time
#
# import _init_paths
# import cv2
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # from model.faster_rcnn.vgg16 import vgg16
# from model.da_faster_rcnn.resnet import resnet
# from model.da_faster_rcnn.vgg16 import vgg16
#
# # from model.nms.nms_wrapper import nms
# from model.roi_layers import nms
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
# from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
# from model.utils.net_utils import load_net, save_net, vis_detections
# from roi_da_data_layer.roibatchLoader import roibatchLoader
# from roi_da_data_layer.roidb import combined_roidb
# from torch.autograd import Variable
#
# try:
#     xrange  # Python 2
# except NameError:
#     xrange = range  # Python 3
#
# from lib.detection_boxes_tools import detection_boxes
#
#
# args_output_dir = "./"
# args_net = "vgg16"
#
# args_lc = True
# args_gc = True
# args_class_agnostic = False
#
#
#
# def excute(_GPUID,_dataset, _model_dir,_round):
#     np.random.seed(cfg.RNG_SEED)
#     os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPUID)
#     t_imdb_test_name=""
#     print("loading our dataset...........")
#     if _dataset == "cityscapefoggy":
#         t_imdb_test_name = "cityscapefoggy_test"
#         args_set_cfgs = [
#             "ANCHOR_SCALES",
#             "[8,16,32]",
#             "ANCHOR_RATIOS",
#             "[0.5,1,2]",
#             "MAX_NUM_GT_BOXES",
#             "30",
#         ]
#
#     elif _dataset == "clipart":
#         t_imdb_test_name = "clipart_test"
#         args_set_cfgs = [
#             "ANCHOR_SCALES",
#             "[8,16,32]",
#             "ANCHOR_RATIOS",
#             "[0.5,1,2]",
#             "MAX_NUM_GT_BOXES",
#             "20",
#         ]
#
#     elif _dataset == "watercolor":
#         t_imdb_test_name = "watercolor_test"
#         args_set_cfgs = [
#             "ANCHOR_SCALES",
#             "[8,16,32]",
#             "ANCHOR_RATIOS",
#             "[0.5,1,2]",
#             "MAX_NUM_GT_BOXES",
#             "20",
#         ]
#
#     args_cfg_file = (
#         "cfgs/{}_ls.yml".format(args_net)
#         if args_large_scale
#         else "cfgs/{}.yml".format(args_net)
#     )
#
#     if args_cfg_file is not None:
#         cfg_from_file(args_cfg_file)
#     if args_set_cfgs is not None:
#         cfg_from_list(args_set_cfgs)
#
#     print("Using config:")
#     pprint.pprint(cfg)
#
#     cfg.TRAIN.USE_FLIPPED = False
#
#
#     #目标域训练集:test
#     imdb, roidb, ratio_list, ratio_index = combined_roidb(
#         t_imdb_test_name, False
#     )
#
#     imdb.competition_mode(on=True)
#     print("{:d} roidb entries".format(len(roidb)))
#
#     print(_model_dir)
#
#     # initilize the network here.
#     if args_net == "vgg16":
#         fasterRCNN = vgg16(
#             imdb.classes,
#             pretrained=False,
#             pretrained_path=None,
#             class_agnostic=args_class_agnostic,
#             lc=args_lc,
#             gc=args_gc,
#         )
#     elif args_net == "res101":
#         fasterRCNN = resnet(
#             imdb.classes,
#             101,
#             pretrained=False,
#             pretrained_path=None,
#             class_agnostic=args_class_agnostic,
#             lc=args_lc,
#             gc=args_gc,
#         )
#     elif args_net == "res50":
#         fasterRCNN = resnet(
#             imdb.classes, 50, pretrained=False, class_agnostic=args_class_agnostic
#         )
#     elif args_net == "res152":
#         fasterRCNN = resnet(
#             imdb.classes, 152, pretrained=False, class_agnostic=args_class_agnostic
#         )
#     else:
#         print("network is not defined")
#         pdb.set_trace()
#
#     fasterRCNN.create_architecture()
#
#     print("load checkpoint %s" % (_model_dir))
#     checkpoint = torch.load(_model_dir)
#     fasterRCNN.load_state_dict(
#         {k: v for k, v in checkpoint["model"].items() if k in fasterRCNN.state_dict()}
#     )
#     # fasterRCNN.load_state_dict(checkpoint['model'])
#     if "pooling_mode" in checkpoint.keys():
#         cfg.POOLING_MODE = checkpoint["pooling_mode"]
#
#     print("load model successfully!")
#
#     start = time.time()
#     all_boxes=detection_boxes.get_test_boxes(imdb, roidb, ratio_list, ratio_index,fasterRCNN)
#
#     #对目标域 test 检测并存储结果
#     imdb.evaluate_detections(all_boxes, args_output_dir, _round,False)
#     end = time.time()
#     print("测试集 检测时间 time: %0.4fs" % (end - start))
#
#     # with open(det_file, "wb") as f:
#     with open("predict_all_boxes.pkl", "wb") as f:
#         pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
#
#     print("Evaluating detections")
#
#     if not os.path.exists(args_output_dir):
#         os.makedirs(args_output_dir)
#     return True
#
#
# if __name__ == "__main__":
#     # excute()
#     pass
#
#     # l=write2list(all_boxes)
#     # return l
