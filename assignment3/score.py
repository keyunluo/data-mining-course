#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-26 下午5:04
# @Author  : 骆克云
# @File    : score.py
# @Software: PyCharm

import numpy as np
from collections import defaultdict
from multiprocessing import Pool
from multiprocessing import cpu_count
import pandas as pd
from functools import reduce


#################################################
######### #### 工具方法   ########################

def distance(Xi, Xj):
    """曼哈顿距离"""
    return sum(np.fabs(Xi - Xj))

def distance_all(data):
    dist = defaultdict(dict)
    indexes = data.index
    for index1 in indexes:
        for index2 in indexes:
            dist[index1][index2] = distance(data.loc[index1].values, data.loc[index2].values)
    return dist


def parallelize_dataframe(df, func):
    df_split = np.array_split(df, cpu_count(), axis=0)
    pool = Pool()
    rets = pool.map(func, df_split)
    pool.close()
    pool.join()
    result = reduce(lambda r, d: r.update(d) or r, rets, {})
    print(result[963][750])
    return result


def distance_parallel(data):
    return parallelize_dataframe(data, distance_all)


#################################################


def score(label_alg, label_true, algorithm="purity"):
    """
    聚类结果评价
    :param label_alg: 算法得出的标签
    :param label_true: 真实的标签
    :param algorithm: 使用的评估算法， 包括"purity"(cluster purity)和"gini"(class-based Gini index)
    :return: score_value
    """
    # 聚类的标签
    clusters_true = label_true.unique()
    clusters_alg = label_alg.unique()
    Ni = {}  # Ni: 本身为第i类实际被分到第j类
    Mj = {}  # Mj: 实际被分到第j类本身为第i类
    Pj = {}  # Pj: 主导类
    Gj = {}  # Gj: Gini系数
    # m: 存储本身为第i类实际被分到第j类的个数
    m = defaultdict(dict)
    for cluster in clusters_true:
        label_true_index = label_true[label_true == cluster].index
        label_j = label_alg[label_true_index]
        for j in clusters_true:
            m[cluster][j] = label_j[label_j == j].count()
        Ni[cluster] = sum(m[cluster][j] for j in clusters_alg)

    for cluster in clusters_alg:
        Mj[cluster] = sum(m[i][cluster] for i in clusters_true)
        Pj[cluster] = max(m[i][cluster] for i in clusters_true)
        Gj[cluster] = 1 - sum(np.square(m[i][cluster]*1.0/Mj[cluster]) for i in clusters_true)

    purity_value = 1.0 * sum(Pj[j] for j in clusters_alg) / sum(Mj[j] for j in clusters_alg)
    gini_value = 1.0 * sum(Gj[j]*Mj[j] for j in clusters_alg) / sum(Mj[j] for j in clusters_alg)

    return purity_value, gini_value
