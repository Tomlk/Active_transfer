"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import, division, print_function

import datasets
import numpy as np
import PIL
from lib.datasets.factory import get_imdb
from lib.model.utils.config import cfg


def prepare_roidb(imdb):
    """
    处理数据集：获取数据集的信息（图片长宽，位置，max_overlaps、max_classes）
    Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

    roidb = imdb.roidb
    if (
        not (imdb.name.startswith("coco"))
        or "car" in imdb.name
        or "sim10k" in imdb.name
    ):
        sizes = [
            PIL.Image.open(imdb.image_path_at(i)).size for i in range(imdb.num_images)
        ]

    for i in range(len(imdb.image_index)):
        roidb[i]["img_id"] = imdb.image_id_at(i)
        roidb[i]["image"] = imdb.image_path_at(i)
        if (
            not (imdb.name.startswith("coco"))
            or "car" in imdb.name
            or "sim10k" in imdb.name
        ):
            roidb[i]["width"] = sizes[i][0]
            roidb[i]["height"] = sizes[i][1]
        # need gt_overlaps as a dense array for argmax
        gt_overlaps = roidb[i]["gt_overlaps"].toarray()

        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        roidb[i]["max_classes"] = max_classes
        roidb[i]["max_overlaps"] = max_overlaps
        # sanity checks
        # max overlap of 0 => class should be zero (background)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # max overlap > 0 => class should not be zero (must be a fg class)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    '''

    Args:
        roidb: roidb数据集

    Returns:
        排序后的ratio_list
        原始list每个位置的排名
    '''
    # rank roidb based on the ratio between width and height.
    ratio_large = 2  # largest ratio to preserve.
    ratio_small = 0.5  # smallest ratio to preserve.

    ratio_list = []
    for i in range(len(roidb)):
        width = roidb[i]["width"]
        height = roidb[i]["height"]
        ratio = width / float(height)

        if ratio > ratio_large:
            roidb[i]["need_crop"] = 1
            ratio = ratio_large
        elif ratio < ratio_small:
            roidb[i]["need_crop"] = 1
            ratio = ratio_small
        else:
            roidb[i]["need_crop"] = 0

        ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list) # 得到ratio_list中每个大小的排名
    return ratio_list[ratio_index], ratio_index


def filter_roidb(roidb):
    # filter the image without bounding box.
    print("before filtering, there are %d images..." % (len(roidb)))
    i = 0
    while i < len(roidb):
        if len(roidb[i]["boxes"]) == 0:
            del roidb[i]
            i -= 1
        i += 1

    print("after filtering, there are %d images..." % (len(roidb)))
    return roidb


def combined_roidb(imdb_names, training=True):  # dataset name
    '''

    Args:
        imdb_names: 数据集名称
        training: 是否训练集

    Returns:
        imdb:imdb类：数据集信息，包括box
        roidb:数据集属性
        ratio_list：排序后的ration 列表
        ratio_index:原始ratio位置的排名
    '''
    def get_training_roidb(imdb):
        """Returns a roidb (Region of Interest database) for use in training."""
        if cfg.TRAIN.USE_FLIPPED:
            print("Appending horizontally-flipped training examples...")
            imdb.append_flipped_images()  #  data augment
            print("done")

        print("Preparing training data...")

        prepare_roidb(imdb)
        # ratio_index = rank_roidb_ratio(imdb)
        print("done")

        return imdb.roidb

    def get_roidb(imdb_name):
        '''
        根据数据集名称
        Args:
            imdb_name:
        Returns:
            roidb
        '''
        imdb = get_imdb(
            imdb_name
        )  # return a pascal_voc dataset object     get_imdb is from factory which contain all legal dataset object

        print("Loaded dataset `{:s}` for training".format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print("Set proposal method: {:s}".format(cfg.TRAIN.PROPOSAL_METHOD))

        roidb = get_training_roidb(imdb)
        return roidb

    print(imdb_names)
    roidbs = [get_roidb(s) for s in imdb_names.split("+")]

    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split("+")[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
        imdb._image_index = (
            get_imdb(imdb_names.split("+")[1]).image_index
            + get_imdb(imdb_names.split("+")[0]).image_index
        )
    else:
        imdb = get_imdb(imdb_names)

    if training:
        roidb = filter_roidb(roidb)  # filter samples without bbox
        print(len(roidb))

    ratio_list, ratio_index = rank_roidb_ratio(roidb)
    return (
        imdb,
        roidb,
        ratio_list,
        ratio_index,
    )  # dataset, roidb dict,ratio_list(0.5,0.5,0.5......2,2,2,), ratio_increase_index(4518,6421,.....)