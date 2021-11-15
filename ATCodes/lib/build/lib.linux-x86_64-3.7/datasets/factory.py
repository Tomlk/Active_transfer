# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import, division, print_function

import numpy as np
from datasets.cityscape import cityscape
from datasets.cityscapefoggy import cityscapefoggy
from datasets.cityscapes_car import cityscapes_car
from datasets.clipart import clipart
from datasets.bdddaytime8 import bdddaytime8
from datasets.bdddaytime10 import bdddaytime10
from datasets.bddnight10 import bddnight10
from datasets.sim10k import sim10k
from datasets.cityscape10k import cityscape10k

from datasets.coco import coco
from datasets.imagenet import imagenet
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.rpc import rpc
from datasets.rpc_fake import rpc_fake
from datasets.vg import vg
# from datasets.water import water
from datasets.watercolor import watercolor

__sets = {}

print("sets:")
print(__sets)

# Set up voc_<year>_<split>
for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc(split, year)

for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_water(split, year)

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name="bdddaytime8_{}".format(split)
        __sets[name]=lambda split=split:bdddaytime8(split,year)

for year in ["2007"]:
    for split in ["trainval", "train", "test"]:
        name="bdddaytime10_{}".format(split)
        __sets[name]=lambda split=split:bdddaytime10(split,year)

for year in ["2007"]:
    for split in ["trainval","train","test"]:
        name="bddnight10_{}".format(split)
        __sets[name]=lambda split=split:bddnight10(split,year)


for year in ["2007"]:
    for split in ["trainval", "test"]:
        name = "watercolor_{}".format(split)
        __sets[name] = lambda split=split: watercolor(split, year)

for year in ["2007", "2012"]:
    for split in ["trainval", "test"]:
        name = "cityscape_{}".format(split)
        __sets[name] = lambda split=split, year=year: cityscape(split, year)

for year in ["2007", "2012"]:
    for split in ["trainval", "test"]:
        name = "cityscapefoggy_{}".format(split)
        __sets[name] = lambda split=split, year=year: cityscapefoggy(split, year)

for year in ["2007", "2012"]:
    for split in ["trainval", "test","train"]:
        name = "sim10k_{}".format(split)
        __sets[name] = lambda split=split, year=year: sim10k(split, year)

for year in ["2007", "2012"]:
    for split in ["trainval", "test","train"]:
        name = "cityscape10k_{}".format(split)
        __sets[name] = lambda split=split, year=year: cityscape10k(split, year)

for year in ["2007"]:
    for split in ["val", "test"]:
        name = "rpc_{}".format(split)
        __sets[name] = lambda split=split: rpc(split, year)

for year in ["2007"]:
    for split in [
        "train",
    ]:
        name = "rpc_fake_{}".format(split)
        __sets[name] = lambda split=split: rpc_fake(split, year)


# Set up coco_2014_<split>
for year in ["2014"]:
    for split in ["train", "val", "minival", "valminusminival", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2014_cap_<split>
for year in ["2014"]:
    for split in ["train", "val", "capval", "valminuscapval", "trainval"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up coco_2015_<split>
for year in ["2015"]:
    for split in ["test", "test-dev"]:
        name = "coco_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: coco(split, year)

# Set up sim10k coco style and cityscapes coco style
for year in ["2019"]:
    for split in ["train", "val"]:
        name = "cityscapes_car_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: cityscapes_car(split, year)


# Set up vg_<split>
# for version in ['1600-400-20']:
#     for split in ['minitrain', 'train', 'minival', 'val', 'test']:
#         name = 'vg_{}_{}'.format(version,split)
#         __sets[name] = (lambda split=split, version=version: vg(version, split))
for version in [
    "150-50-20",
    "150-50-50",
    "500-150-80",
    "750-250-150",
    "1750-700-450",
    "1600-400-20",
]:
    for split in [
        "minitrain",
        "smalltrain",
        "train",
        "minival",
        "smallval",
        "val",
        "test",
    ]:
        name = "vg_{}_{}".format(version, split)
        __sets[name] = lambda split=split, version=version: vg(version, split)

# set up image net.
for split in ["train", "val", "val1", "val2", "test"]:
    name = "imagenet_{}".format(split)
    devkit_path = "data/imagenet/ILSVRC/devkit"
    data_path = "data/imagenet/ILSVRC"
    __sets[
        name
    ] = lambda split=split, devkit_path=devkit_path, data_path=data_path: imagenet(
        split, devkit_path, data_path
    )


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())