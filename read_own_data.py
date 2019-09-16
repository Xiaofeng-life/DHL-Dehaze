#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# @Time    : 2019/9/12 15:25 
# @Author  : CongXiaofeng 
# @File    : read_own_data.py 
# @Software: PyCharm

import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils import data
import matplotlib.pyplot as plt


def load_data(basic_dir, batch_size=4, mode="train", image_size=(32, 32)):
    transform = []
    # 训练模式下进行随机翻转
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    # 归一化操作
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)
    dataset = ImageFolder(basic_dir, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return data_loader

if __name__ == "__main__":
    batch_size = 4
    data_loader = load_data(basic_dir="own_data/", batch_size=batch_size, mode="test")
    # 实现迭代器协议， 就可以遍历data loader
    data_loader = iter(data_loader)
    image, label = next(data_loader)
    # 将数据先转为numpy，再转置
    # pytorch默认为（batch_size, channel, H, W）
    # 需要转为 （batch_size, H, W, channel）
    image = image.numpy().transpose(0, 2, 3, 1)
    plt.figure()
    for i in range(batch_size):
        plt.subplot(1, batch_size, i+1)
        # image 的值范围是(-1,1)之间
        # 需要转为(0,1)之间显示
        plt.imshow((image[i, :, :, :]+1)/2)
    plt.show()
