#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-17 下午5:24
# @Author  : 骆克云
# @File    : kfolddataset.py
# @Software: PyCharm

import numpy as np


def train_test_split(data, train_size=0.8):
    """测试集/训练集数据划分"""
    length = len(data)
    train_lenth = int(train_size * length)
    m = np.arange(length)
    np.random.shuffle(m)
    train = m[:train_lenth]
    test = m[train_lenth:]
    return train, test


def k_fold_train_test_split(k, data):
    """K折交叉验证：训练集/测试集数据划分"""
    m = np.arange(len(data))
    np.random.shuffle(m)
    train_test_slices = np.array_split(m, k)
    slices_array = np.zeros(0)

    train_test_data = [[] for i in [0, 1]]
    for i in range(k):
        """测试集数据"""
        slices = train_test_slices.copy()
        for j in range(k):
            if j != i:
                slices_array = np.append(slices_array, slices[j]) 
        train_test_data[1].append(np.array(slices[i]).astype(int))
        train_test_data[0].append(slices_array.astype(int))
    return train_test_data
