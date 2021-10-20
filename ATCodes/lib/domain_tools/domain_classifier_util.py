import numpy as np
import torch
import torch.nn.functional as F
from model.da_faster_rcnn_instance_da_weight.faster_rcnn import _fasterRCNN
from roi_da_data_layer.roibatchLoader import roibatchLoader


class Domain_classifier:
    def __init__(self, faster_rcnn: _fasterRCNN, dataset: roibatchLoader):
        '''

        Args:
            faster_rcnn: 带有域分类器的faster-rcnn网络
            dataset:输入的数据集类(包括一些信息)
        '''
        self.my_faster_rcnn = faster_rcnn
        self.dataset = dataset
        self.im_cls_lb = None
        self.num_boxes = None
        self.gt_boxes = None


        self.phi = 0.8  # 系数比
        self.dic = {}  # key:图片名 value:不确定值

    def set_args(self, num_boxes, gt_boxes, da_weight):
        '''
        带有域分类器的faster-rcnn网络需要的一些参数
        Args:
            num_boxes: boxes数目
            gt_boxes: boxes分布情况
            da_weight: 域适应权重

        Returns:

        '''
        self.num_boxes = num_boxes
        self.gt_boxes = gt_boxes
        self.da_weight = da_weight

    def get_args(self, data_im_info, cuda_flag: bool):
        '''

        Args:
            data_im_info: 单个img的图像信息对象
            cuda_flag: 是否转换为cuda数据

        Returns:
            带有域分类器的faster-rcnn网络需要的一些参数
        '''
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

    def get_calculate_domain_list(self, target_flag):
        '''

        Args:
            target_flag: 是否是目标域数据

        Returns: 经过排序后的列表

        '''
        roidb = self.dataset._roidb
        for item in roidb:
            img_id = item["img_id"]
            img_path = item["image"]
            data_im_info = self.dataset.__getitem__(img_id)
            im_data,im_info,im_cls_lb=self.get_args(data_im_info,True)
            out_d_pixel, out_d, _ = self.my_faster_rcnn.forward(  #
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
        '''
            将目标域中目标域特征明显的图片迁移到源域中；
            将源于中源域特征明显的图片移除
        '''
        sorted_tuple = sorted(self.dic.items(), key=lambda d: d[1], reverse=True)

        l=[]
        for item in sorted_tuple:
            l.append(item[0])
        return l

    def get_global_classifier_prob(self, out_d, target_flag):
        '''

        Args:
            out_d: 全局域分类占比结果
            target_flag: 目标域flag

        Returns:
            通过全局域分类器该域的概率
        '''
        P = F.softmax(out_d)
        P_numpy = P.cpu().detach().numpy()[0]
        if not target_flag:  # source
            return P_numpy[0]
        else:  # target
            return P_numpy[1]

    def get_local_classifier_prob(self, out_d_pixel, target_flag):
        '''

        Args:
            out_d_pixel: 局部域分类器的概率列表
            target_flag: 目标域flag

        Returns:
            通过局部域分类器该域的概率
        '''
        loss_source_value = 0.5 * torch.mean(out_d_pixel ** 2)
        loss_target_value = 0.5 * torch.mean((1 - out_d_pixel) ** 2)
        out_d_pixels = torch.tensor([[loss_target_value, loss_source_value]])
        P = F.softmax(out_d_pixels)
        P_numpy = P.cpu().detach().numpy()[0]
        if not target_flag:  # source
            return P_numpy[0]
        else:  # target
            return P_numpy[1]

