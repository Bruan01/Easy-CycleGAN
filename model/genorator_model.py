# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 11:10
@Author  : BruanLin
@File    : genorator_model.py
#  生成器
'''

import torch
import torch.nn as nn


# 卷积块
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        """
        初始化卷积块
        :param in_channels:
        :param out_channels:
        :param down: 是否下采样
        :param use_act: 激活？
        :param kwargs:
        """
        super().__init__()
        self.conv = nn.Sequential(
            # 下采样就使用卷积,如果上采样就实行转置卷积
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            # InstanceNorm: 一个channel内做归一化, 算H * W的均值, 用在风格化迁移;
            # 因为在图像风格化中, 生成结果主要依赖于某个图像实例, 所以对整个batch归一化不适合
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
            # identity模块不改变输入，直接return input
            # 一种编码技巧吧，比如我们要加深网络，有些层是不改变输入数据的维度的，
            # 在增减网络的过程中我们就可以用identity占个位置，这样网络整体层数永远不变，
        )

    def forward(self, x):
        return self.conv(x)


# 残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            # 参数是按照论文实现的
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_residuals=9):
        super(Generator, self).__init__()
        # 初始块
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        # 下采样块
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1)
            ]
        )
        # 残差块，拆包 九个串在一起
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        # 上采样
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features * 4, num_features * 2, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
                ConvBlock(num_features * 2, num_features * 1, down=False, kernel_size=3, stride=2, padding=1,
                          output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)  # 初始
        for layer in self.down_blocks:  # 下采样 提取图像信息
            x = layer(x)
        x = self.residual_blocks(x)  # 残差块卷积
        for layer in self.up_blocks:  # 上采样 还原图像信息
            x = layer(x)
        return torch.tanh(self.last(x))  # 激活


# 测试模块
def test():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test()
