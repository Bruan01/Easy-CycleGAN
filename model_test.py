# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 11:21
@Author  : BruanLin
@File    : model_test.py

'''
from model import discriminator_model

if __name__ == '__main__':
    D_conv = discriminator_model.Block(10,5,1)
    print(D_conv)

