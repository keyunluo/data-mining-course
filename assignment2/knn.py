#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-30 下午7:17
# @Author  : 骆克云
# @File    : knn.py
# @Software: PyCharm

import numpy as np


def knn(train_data, train_label, test_data, k=1):
    """
    knn:分类
    :param train_data:训练数据集
    :param train_label:训练数据集标签
    :param test_data:测试数据集
    :param k:近邻的个数
    :return:测试数据集标签
    """

    row_test, _ = test_data.shape
    row_train, _ = train_data.shape
    labels = []

    for row_index_test in range(row_test):
        dis_row = {}
        label = {}
        for row_index_train in range(row_train):
            dis_row[row_index_train] = np.linalg.norm(test_data[row_index_test]-train_data[row_index_train])
        dis_sorted = sorted(dis_row.items(), key=lambda d: d[1])[:k]
        # 对距离从小到大排序，对标签累加后取最大的一个
        for dis in dis_sorted:
            label_train = train_label[dis[0]]
            label[label_train] = label.get(label_train, 0) + 1
        label_sort = sorted(label.items(), key=lambda d: d[1])[-1]
        labels.append(label_sort[0])
    return labels
