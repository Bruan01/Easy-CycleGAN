# Easy-CycleGAN

## 模型构造

模型基于论文简化的Pytorch实现，其中有一些简单修改，对比项目https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
做了简化，删去了很多东西，因为我看不懂，我是菜逼。
![img.png](/static/img.png)
![img_1.png](/static/img_1.png)

# 手法

没有什么手法，就是乱写的，用到Pytorch,其他也没什么了，要有个好一点的显卡，这家伙跑起来贼慢

## 关于数据集

1. 数据集文件格式  
─dataset  
│  ├─monet2photo    
│  │  ├─test  
│  │  │  ├─testA  
│  │  │  └─testB  
│  │  └─train  
│  │      ├─trainA    
│  │      └─trainB


## 部署

1. 直接copy下来即可
2. 配置文件修改 [修改数据集文件路径]  
`TRAIN_DIR = "dataset/monet2photo/train"`  
`VAL_DIR = "dataset/monet2photo/test"`
3. 使用配置

| pytorch | 1.12.1 |
|---------|--------|
| python  | 3.8.13 |
| 其他      |        |


## train
看好路径，买块好显卡，准备好数据集，直接train就行！炼丹！


## test 
还没写完。。