#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-26 下午4:54
# @Author  : 骆克云
# @File    : spectralclustering.py
# @Software: PyCharm

from projectutil import get_handwritten_digits
from scipy.spatial.distance import cdist
from .kmedoids import KMedoids
import numpy as np


class Spectral:
    def __init__(self, filename, knn=3):
        handwritten_digits, true_label = get_handwritten_digits(filename)
        self.filename = filename
        self.true_label = true_label
        self.k = len(true_label.unique())
        self.length = len(handwritten_digits)
        self.distance = cdist(handwritten_digits, handwritten_digits, metric="euclidean")
        self.knn = knn

    def build_graph_weights(self):
        k = self.knn + 1
        # 图矩阵,初始化为0
        N = self.length
        W = np.full((N, N), fill_value=0, dtype=float)

        # 计算K近邻,并且对称化
        for i in range(N):
            index_sort = np.argsort(self.distance[i])
            for j in index_sort[:k]:
                W[i, j] = 1
            W[i, i] = 0

        return np.maximum(W, W.T)

    def build_matrix(self):
        W = self.build_graph_weights()
        N = self.length
        D = np.full((N, N), fill_value=0, dtype=float)
        for i in range(N):
            D[i, i] = sum(W[i][j] for j in range(N))
        #L = D - W
        #L = np.eye(N) - np.linalg.inv(D).dot(W)
        D_sym = np.full((N, N), fill_value=0, dtype=float)
        for i in range(N):
            D_sym[i, i] = np.power(D[i, i], -0.5)
        L_sym = np.eye(N) - D_sym.dot(W).dot(D_sym)
        return L_sym

    def laplas(self):
        L = self.build_matrix()
        eig_values, eig_vectors = np.linalg.eigh(L)
        # 对特征值，特征向量排序
        indexes = np.argsort(eig_values)
        eig_vectors = eig_vectors[:, indexes]
        return eig_vectors[:, :self.k]


    def run(self):
        L = self.laplas()
        kmedoids = KMedoids(self.filename, repeat=4, alg="Spectral-Clustering-knn-%d" % self.knn)
        kmedoids.reinit(L)
        kmedoids.run()
