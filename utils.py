# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 15:22
@Author  : BruanLin
@File    : utils.py

'''
import random, torch, os, numpy as np
import config


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    # 保存模型和优化器的权重文件
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    # 加载模型权重
    print("=> Loading Checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    # 如果不加上这个 优化器的学习率速度会改变，跟原来不一样
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr



