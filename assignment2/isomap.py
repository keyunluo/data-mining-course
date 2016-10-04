#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-28 下午8:40
# @Author  : 骆克云
# @File    : isomap.py
# @Software: PyCharm

from projectutil import get_reduction_dataset
import numpy as np
from .knn import onenn


class ISOMAP:
    """ISOMAP流式学习降维算法"""
    def __init__(self, name="sonar", k_nn=4):
        """初始化：输入文件名：sonar/splice"""
        data_train, label_train = get_reduction_dataset(name, "train")
        data_test, label_test = get_reduction_dataset(name, "test")
        self.name = name
        self.k_nn = k_nn
        self.train = np.array(data_train).astype(float)
        self.train_label = np.array(label_train).astype(int)
        self.test = np.array(data_test).astype(float)
        self.test_label = np.array(label_test).astype(int)
        self.data = np.append(self.train, self.test, axis=0)
        self.length = len(self.data)

    def distance_matrix(self):
        """计算原始点之间的距离矩阵"""
        dis_points = np.zeros((self.length, self.length))
        for i, line_i in enumerate(self.data):
            for j, line_j in enumerate(self.data):
                dis_points[i, j] = np.linalg.norm(line_i-line_j) if i != j else 0
        return dis_points

    def build_graph(self):
        """构建图模型：每个点和其周围最近的k个点有连接"""
        # 点到自身的距离为0,故k+=1
        k = self.k_nn + 1
        N = self.length
        dis_matrix = self.distance_matrix()
        # 图矩阵,初始化为无穷大
        graph_matrix = np.full((N, N), fill_value=np.inf, dtype=float)

        # 计算K近邻,并且对称化
        for i in range(N):
            index_sort = np.argsort(dis_matrix[i])
            for j in index_sort[:k]:
                graph_matrix[i, j] = dis_matrix[i, j]

        return np.minimum(graph_matrix, graph_matrix.T)

    def shortest_path(self, algorithm="floyd"):
        """
        floyd 任意两点间距离
        采用numpy外积运算
        """
        graph_matrix = self.build_graph()

        for i in range(self.length):
            graph_matrix = np.minimum(graph_matrix, np.add.outer(graph_matrix[:, i], graph_matrix[i, :]))
        # Floyd算法
        """
        for k in range(self.length):
            for j in range(self.length):
                for i in range(self.length):
                    if graph_matrix[i][k] + graph_matrix[k][j] < graph_matrix[i][j]:
                        graph_matrix[i][j] = graph_matrix[i][k] + graph_matrix[k][j]
        """

        return graph_matrix

    def mds(self):
        """
        经典MDS算法：
        S= -1/2*H*D*H
        H = I - 11^T/N
        :return: eig_vectors:特征向量，eig_values：特征值
        """
        graph_matrix = self.shortest_path()

        # 消除不连接分量
        isINF = np.less(graph_matrix,np.inf)
        MAX = [-np.inf, -1]
        connected = []
        for i in range(len(graph_matrix)):
            conn = np.nonzero(isINF[i])[0]
            connected.append(conn)
            if MAX[0] < len(conn):
                MAX = [len(conn), i]
        connected_index = connected[MAX[1]]
        # 非连通量
        erase_value = [index for index, _ in enumerate(connected_index) if index not in connected_index]
        # 全连通距离矩阵
        conn_graph_matrix = np.take(np.take(graph_matrix, connected_index, 0), connected_index, 1)

        #print(graph_matrix[0])
        N = len(conn_graph_matrix)#self.length

        D = -0.5 * conn_graph_matrix**2
        # 中心矩阵
        H = np.eye(N) - np.ones((N, N)) / N
        S = H.dot(D).dot(H)
        # 对称矩阵特征值分解分解
        eig_values, eig_vectors = np.linalg.eigh(S)
        # 对特征值，特征向量排序
        indexes = np.argsort(- eig_values)
        eig_values = eig_values[indexes]
        eig_vectors = eig_vectors[:, indexes]

        return eig_values, eig_vectors, erase_value

    def projection(self):
        """投影训练集，测试集"""
        projection_trains = []
        projection_tests = []
        eig_values, eig_vectors, erase_value = self.mds()
        # 仅使用正的特征值计算坐标
        indexes, = np.where(eig_values > 0)

        if len(indexes) < 30:
            print("!!!正的特征值不足三十个，无法进行纵向比较!!!")
            return -1, -1
        # 判断非连通量，即孤立噪声点所在的数据集
        train_len = len(self.train)
        erase_train = []
        erase_test = []
        if len(erase_value) != 0:
            for i in erase_value:
                if i < train_len:
                    erase_train.append(i)
                else:
                    erase_test.append(train_len-i)
            train_len -= len(erase_train)
            print("===ISOMAP 文件:%s:出现孤立点===" % self.name)
            print("训练集位置:%s,测试集位置:%s" % (str(erase_train), str(erase_test)))

        for k in [10, 20, 30]:
            U = eig_vectors[:, indexes[:k]]
            delta = np.diag(np.sqrt(eig_values[indexes[:k]]))
            Y = U.dot(delta)
            projection_trains.append(Y[:train_len])
            projection_tests.append(Y[train_len:])

        return projection_trains, projection_tests, erase_train, erase_test

    def run(self):
        project = self.projection()
        projections = project[:2]
        erase_train, erase_test = project[2:]
        if projections[0] != -1:
            train_label = np.delete(self.train_label, erase_train)
            test_label = np.delete(self.test_label, erase_test)
            labels = (train_label, test_label)
            onenn(projections, labels, self.name, "ISOMAP-k"+str(self.k_nn))
