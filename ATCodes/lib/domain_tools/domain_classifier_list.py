import numpy as np
import torch
import torch.nn.functional as F
from model.da_faster_rcnn_instance_da_weight.faster_rcnn import _fasterRCNN
from roi_da_data_layer.roibatchLoader import roibatchLoader
from domain_tools.domain_classifier_util import Domain_classifier


def get_source_target_list(select_strategy,faster_rcnn: _fasterRCNN, dataset_s: roibatchLoader, dataset_t: roibatchLoader,gt_boxes,num_boxes):
    source_list=[]
    target_list=[]
    if select_strategy==2:
        target_list=get_sorted_list(faster_rcnn,dataset_t,gt_boxes,num_boxes,True)
    elif select_strategy==4 or select_strategy==5:
        target_list=get_sorted_list(faster_rcnn,dataset_t,gt_boxes,num_boxes,True)
        source_list=get_sorted_list(faster_rcnn,dataset_s,gt_boxes,num_boxes,False)

    for i in range(len(source_list)):
        source_list[i] = (source_list[i].split("/"))[-1]
    for i in range(len(target_list)):
        target_list[i] = (target_list[i].split("/"))[-1]

    return source_list,target_list


def get_sorted_list(faster_rcnn: _fasterRCNN, dataset: roibatchLoader,gt_boxes,num_boxes,target_flag):
    DC_source = Domain_classifier(faster_rcnn, dataset)
    gt_boxes.data.resize_(1, 1, 5).zero_()
    num_boxes.data.resize_(1).zero_()
    DC_source.set_args(
        num_boxes,
        gt_boxes,
        da_weight=1.0,
    )
    result_list = DC_source.get_calculate_domain_list(target_flag)
    return result_list