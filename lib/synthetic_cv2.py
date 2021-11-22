#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.base_dataset import BaseDataset


labels_info = [
    {"hasInstances": False, "category": "sky", "catid": 0, "name": "sky", "ignoreInEval": False, "id": 0, "color": [0, 0, 0], "trainId": 0},
    {"hasInstances": False, "category": "ground", "catid": 1, "name": "soil", "ignoreInEval": False, "id": 1, "color": [111, 74, 0], "trainId": 1},
    {"hasInstances": False, "category": "ground", "catid": 1, "name": "trails", "ignoreInEval": False, "id": 2, "color": [81, 0, 81], "trainId": 2},
    {"hasInstances": False, "category": "vegatation", "catid": 2, "name": "tree canopy", "ignoreInEval": False, "id": 3, "color": [128, 64, 128], "trainId": 3},
    {"hasInstances": False, "category": "vegatation", "catid": 2, "name": "fuel", "ignoreInEval": False, "id": 4, "color": [244, 35, 232], "trainId": 4},
    {"hasInstances": False, "category": "vegatation", "catid": 2, "name": "trunks", "ignoreInEval": False, "id": 5, "color": [250, 170, 160], "trainId": 5},
    {"hasInstances": False, "category": "vegatation", "catid": 2, "name": "stumps", "ignoreInEval": False, "id": 6, "color": [0, 170, 160], "trainId": 6},
]



class Synthetic(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(Synthetic, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 7
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb TODO try to find values for outdoor / our dataset
            std=(0.2112, 0.2148, 0.2115),
        )





if __name__ == "__main__":
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    ds = Synthetic('./data/', mode='val')
    dl = DataLoader(ds,
                    batch_size = 4,
                    shuffle = True,
                    num_workers = 4,
                    drop_last = True)
    for imgs, label in dl:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
