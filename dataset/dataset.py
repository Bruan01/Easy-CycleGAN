# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 11:11
@Author  : BruanLin
@File    : dataset.py
关于数据集的导入测试
'''
import torch
from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """ 数据集 基础类！"""

    def __init__(self, root_a, root_b, transform=None):
        self.root_a = root_a,
        self.root_b = root_b,


class Horse2ZebraDataset(Dataset):
    def __init__(self, root_zebra, root_horse, trans_forms=None):
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = trans_forms

        # 查看数据集目录
        self.zebra_images = os.listdir(root_zebra) # 打开文件夹
        self.horse_images = os.listdir(root_horse)
        # 由于数据集长度不一样，这里选要做出处理（选择上界）
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))  # 1000 > 1500(^)
        # 得到各自长度
        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, item):
        # 得到图片索引
        zebra_img = self.zebra_images[item % self.zebra_len]
        horse_img = self.horse_images[item % self.horse_len]
        # 拼接具体路径
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)
        # 将图片读入
        zebra_img = np.array(Image.open(zebra_path).convert("RGB"))
        horse_img = np.array(Image.open(horse_path).convert("RGB"))
        # 是否进行图像增强
        if self.transform:
            augmentations = self.transform(image=zebra_img, image0=horse_img)
            zebra_img = augmentations["image"]
            horse_img = augmentations["image0"]

        return zebra_img,horse_img
