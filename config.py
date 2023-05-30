# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 15:22
@Author  : BruanLin
@File    : config.py
保存参数
'''
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# albumentations 是一个给予 OpenCV的快速训练数据增强库，
# 拥有非常简单且强大的可以用于多种任务（分割、检测）的接口，易于定制且添加其他框架非常方便。
# 它可以对数据集进行逐像素的转换，如模糊、下采样、高斯造点、高斯模糊、动态模糊、RGB转换、随机雾化等；
# 也可以进行空间转换（同时也会对目标进行转换），如裁剪、翻转、随机裁剪等。

# 超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/monet2photo/train"
VAL_DIR = "dataset/monet2photo/test"
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200
LOAD_MODEL = False
SAVE_MODEL = False
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tra"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"

# 图像转换
transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5), # 垂直翻转
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255), # 归一化
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
