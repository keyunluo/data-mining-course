#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-28 下午8:13
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm

from assignment2.pca import PCA
from assignment2.svd import SVD


def run():
    """
    # sonar文件
    print("sonar文件测试结果：")
    print("PCA算法：")
    pca = PCA("sonar")
    pca.onenn()

    # splice文件
    print("splice文件测试结果：")
    print("PCA算法：")
    pca = PCA("splice")
    pca.onenn()
    """
    pca = PCA("sonar")
    pca.onenn()
    pca.onenn_use_knn()
    #svd = SVD("sonar")

    #svd.onenn()
    #svd = SVD("splice")
    #svd.onenn()

