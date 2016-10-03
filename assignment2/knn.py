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


def onenn(projections, labels, name, algorithm):
    """
    最近邻计算
    :param projections: projection_trains, projection_tests元组
    :param labels: train_label, test_label
    :param name: 文件名
    :param algorithm: 使用的算法
    :return: 无
    """

    print("====算法：%s ==== \n%s文件测试结果：" % (algorithm, name))
    projection_trains, projection_tests = projections
    train_label, test_label = labels

    # 分三种情况(k=10,20,30)计算正确率
    for i in range(3):

        row_test, _ = projection_tests[i].shape
        row_train, _ = projection_trains[i].shape
        count = 0
        for row_index in range(row_test):
            dist_row = []
            for rowtrain in range(row_train):
                # 欧式距离
                dist_row.append(np.linalg.norm(projection_tests[i][row_index] - projection_trains[i][rowtrain]))
            # 取距离最短的点的标签
            result = train_label[dist_row.index(min(dist_row))]

            if result == test_label[row_index]:
                count += 1

        # 输出结果
        if i == 0:
            print("k=10,正确率：{0}/{1}={2}%".format(count, row_test, 1.0 * count / row_test * 100))

        if i == 1:
            print("k=20,正确率：{0}/{1}={2}%".format(count, row_test, 1.0 * count / row_test * 100))

        if i == 2:
            print("k=30,正确率：{0}/{1}={2}%".format(count, row_test, 1.0 * count / row_test * 100))
