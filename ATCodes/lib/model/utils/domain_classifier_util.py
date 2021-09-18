import numpy as np
import torch
import torch.nn.functional as F
from lib.model.da_faster_rcnn_instance_da_weight.faster_rcnn import _fasterRCNN
from lib.roi_da_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.transfer_util import transfer_util


class Domain_classifier:
    def __init__(self, faster_rcnn: _fasterRCNN, dataset: roibatchLoader, extracted_num: int):
        self.my_faster_rcnn = faster_rcnn
        self.dataset = dataset
        self.im_cls_lb = None
        self.num_boxes = None
        self.gt_boxes = None

        self.extracted_num = extracted_num

        self.phi = 0.75  # 系数比
        self.dic = {}  # key:图片名 value:不确定值

    def set_args(self, num_boxes, gt_boxes, da_weight):
        self.num_boxes = num_boxes
        self.gt_boxes = gt_boxes
        self.da_weight = da_weight

    def get_args(self, data_im_info, cuda_flag: bool):
        im_data = data_im_info[0]
        im_info = data_im_info[1]
        im_cls_lb = data_im_info[2]

        #
        im_data = im_data[None, :]
        im_info = im_info[None, :]
        im_cls_lb = im_cls_lb[None, :]

        if cuda_flag:
            # 转cuda
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            im_cls_lb = im_cls_lb.cuda()

        return (im_data, im_info, im_cls_lb)

    def calculate_domain(self, target_flag):
        roidb = self.dataset._roidb
        for item in roidb:
            img_id = item["img_id"]
            img_path = item["image"]
            data_im_info = self.dataset.__getitem__(img_id)
            im_data,im_info,im_cls_lb=self.get_args(data_im_info)
            out_d_pixel, out_d, _ = self.my_faster_rcnn.forward(  # 目标域获得
                im_data,
                im_info,
                im_cls_lb,
                self.gt_boxes,
                self.num_boxes,
                target=True,
                weight_value=self.da_weight,
            )

            global_prob = self.get_global_classifier_prob(out_d, target_flag)
            local_prob = self.get_local_classifier_prob(out_d_pixel, target_flag)
            self.dic[img_path] = self.phi * global_prob + (1 - self.phi) * local_prob
            print(img_id, ":", self.dic[img_path])
            # input()

        sorted_tuple = sorted(self.dic.items(), key=lambda d: d[1], reverse=False)
        extracted_num = min(len(sorted_tuple), self.extracted_num)
        extracted_tuple = sorted_tuple[0:extracted_num]
        for item in extracted_tuple:
            key = item[0]
            print(key, ":", self.dic[key])
        input()

        a_transfer_util = transfer_util(extracted_tuple)

        if target_flag is True:  # target
            a_transfer_util.transfer_files(sorted_tuple)
        else:  # source
            a_transfer_util.remove_files(sorted_tuple)

    def get_global_classifier_prob(self, out_d, target_flag):
        P = F.softmax(out_d)
        P_numpy = P.cpu().detach().numpy()[0]
        if not target_flag:  # source
            return P_numpy[0]
        else:  # target
            return P_numpy[1]

    def get_local_classifier_prob(self, out_d_pixel, target_flag):
        source_value = 0.5 * torch.mean(out_d_pixel ** 2)
        target_value = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
        out_d_pixels = torch.tensor([[source_value, target_value]])
        P = F.softmax(out_d_pixels)
        P_numpy = P.cpu().detach().numpy()[0]
        if not target_flag:  # source
            return P_numpy[0]
        else:  # target
            return P_numpy[1]
