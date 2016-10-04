#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-3 下午6:02
# @Author  : 骆克云
# @File    : DimReductionwithSklearn.py
# @Software: PyCharm

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import Isomap
import numpy as np
from projectutil import get_reduction_dataset
from .knn import onenn


class Reduction:
    """降维：使用Python的Scikit-learn库"""

    def __init__(self, name="sonar"):
        data_train, label_train = get_reduction_dataset(name, "train")
        data_test, label_test = get_reduction_dataset(name, "test")
        self.name = name
        self.train = np.array(data_train).astype(float)
        self.train_label = np.array(label_train).astype(int)
        self.test = np.array(data_test).astype(float)
        self.test_label = np.array(label_test).astype(int)

    def pca_solver(self):
        """PCA降维"""
        trains = []
        tests = []
        for k in [10, 20, 30]:
            pca = PCA(n_components=k)
            trains.append(pca.fit_transform(self.train))
            tests.append(pca.transform(self.test))
        labels = (self.train_label, self.test_label)
        projections = (trains, tests)
        onenn(projections, labels, self.name, "PCA-Scikit-learn")

    def svd_solver(self):
        """SVD降维"""
        trains = []
        tests = []
        for k in [10, 20, 30]:
            svd = TruncatedSVD(n_components=k, n_iter=20)
            trains.append(svd.fit_transform(self.train))
            tests.append(svd.transform(self.test))
        labels = (self.train_label, self.test_label)
        projections = (trains, tests)
        onenn(projections, labels, self.name, "SVD-Scikit-learn")

    def isomap_solver(self, k_nn=4):
        """isomap流式学习降维"""
        trains = []
        tests = []
        data = np.append(self.train, self.test, axis=0)
        for k in [10, 20, 30]:
            isomap = Isomap(n_components=k, n_neighbors=k_nn)
            fited_data = isomap.fit_transform(data)
            trains.append(fited_data[:len(self.train)])
            tests.append(fited_data[len(self.train):])
        labels = (self.train_label, self.test_label)
        projections = (trains, tests)
        onenn(projections, labels, self.name, "ISOMAP-Scikit-learn-k%d" % k_nn)
