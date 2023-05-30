# -*- coding: utf-8 -*-
'''
@Time    : 2023/5/30 11:10
@Author  : BruanLin
@File    : discriminator_model.py
鉴别器模块

'''
import torch
import torch.nn as nn


# 新建一个块,用于下面的鉴别器的实现
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        # 建立一个卷积层
        self.conv = nn.Sequential(
            # 卷积的参数都是论文中提到的，填充模式使用倒映模式，可以减少冲突
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            # InstanceNorm: 一个channel内做归一化, 算H * W的均值, 用在风格化迁移;
            # 因为在图像风格化中, 生成结果主要依赖于某个图像实例, 所以对整个batch归一化不适合
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


# 判别器模块
class Discriminator(nn.Module):
    # 初始化
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        # 初始化
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels=features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2)
        )
        # 用于保存层变量
        layers = []
        in_channels = features[0]
        # 添加块 (第一层用stride1 后面四层stride2)
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature
        # 判别是否为正确图像 [512,1] 卷积输出0-1
        layers.append(nn.Conv2d(in_channels,1,kernel_size=4,stride=1,padding=1,padding_mode="reflect"))
        # 解包
        self.model = nn.Sequential(*layers)

    def forward(self,x):
        x = self.initial(x)
        # 使用sigmod 使其确保在0-1之间
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((1,3,256,256))
    y = torch.randn((1,3,256,256))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape)

if __name__ == "__main__":
    test()