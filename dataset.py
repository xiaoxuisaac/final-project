#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 07:34:12 2022

@author: xiaoxu
"""

import os.path as osp
import os

import torch
from torch_geometric.data import Dataset, Data


class FloquetDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return os.listdir(self.root+"/raw")

    @property
    def processed_file_names(self):
        return os.listdir(self.root+"/processed")



    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            with open(self.root + "/raw/"+ raw_path) as f:
                data = torch.load(f)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'{idx}.all'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'{idx}.all'))
        return data