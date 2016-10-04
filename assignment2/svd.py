#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-28 下午8:40
# @Author  : 骆克云
# @File    : svd.py
# @Software: PyCharm

from projectutil import get_reduction_dataset
import numpy as np
from .knn import onenn

class SVD:
    """
    SVD奇异值分解
    """
    def __init__(self, name="sonar"):
        """初始化：输入文件名：sonar/splice"""
        data_train, label_train = get_reduction_dataset(name, "train")
        data_test, label_test = get_reduction_dataset(name, "test")
        self.name = name
        self.train = np.array(data_train).astype(float)
        self.train_label = np.array(label_train).astype(int)
        self.test = np.array(data_test).astype(float)
        self.test_label = np.array(label_test).astype(int)

    def svd(self):
        """
        奇异值分解：
        A[m*n] = U[m*m]*Sigma[m*n]*Vt[n*n]
        ==>
        A[m*n] = U[m*k]*Sigma[k*k]*Vt[k*n]
        """
        u, sigma, vt = np.linalg.svd(self.train.T)
        # 奇异值对角化
        S = np.zeros(self.train.shape)
        sig_shape = sigma.shape[0]
        S[:sig_shape, :sig_shape] = np.diag(sigma)

        # 按照k=10,20,30保留分解结果
        Us = []
        Sigmas = []
        Vts = []
        for k in [10, 20, 30]:
            Us.append(u[:, :k])
            Sigmas.append(S[:k, :k])
            Vts.append(vt[:k])
        return Us, Sigmas, Vts

    def projection(self):
        """对训练集，测试集进行投影"""
        Us, Sigmas, Vts = self.svd()
        projection_trains = []
        projection_tests = []

        # projection_trains：m*k ; projection_tests：p*k
        for i in range(3):
            U = Us[i]
            Sigma = Sigmas[i]
            Vt = Vts[i]

            projection_trains.append(np.dot(Sigma, Vt).T)
            projection_tests.append(np.dot(U.T, self.test.T).T)

        return projection_trains, projection_tests

    def run(self):
        projections = self.projection()
        labels = (self.train_label, self.test_label)
        onenn(projections, labels, self.name, "SVD")
