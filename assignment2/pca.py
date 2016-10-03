#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-28 下午8:40
# @Author  : 骆克云
# @File    : pca.py
# @Software: PyCharm

from projectutil import get_reduction_dataset
from .knn import knn, onenn
import numpy as np


class PCA:
    """
    PCA算法：处理单一文件
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
        self.meanVal = np.empty(self.train.shape[1])

    def zero_mean(self):
        """求均值，零均值化"""
        self.meanVal = np.mean(self.train, axis=0)
        train_zero = self.train - self.meanVal
        return train_zero

    def cov(self):
        """求协方差矩阵,特征值与特征向量"""
        train_zero = self.zero_mean()
        cov_mat = np.cov(train_zero, rowvar=False)
        eig_values, eig_vectors = np.linalg.eig(cov_mat)
        return eig_values, eig_vectors

    def projectionMatrix(self):
        """计算PCA投影矩阵"""
        eig_values, eig_vectors = self.cov()
        eig_values_index_sort = np.argsort(-eig_values)
        projection_matrix = []
        for k in [10, 20, 30]:
            k_index = eig_values_index_sort[:k]
            projection_matrix.append(eig_vectors[:, k_index])
        return projection_matrix

    def projection(self):
        """投影训练数据，测试数据"""
        projection_matrix = self.projectionMatrix()
        projection_trains = []
        projection_tests = []
        for matrix in projection_matrix:
            projection_trains.append(self.train.dot(matrix))
            projection_tests.append(self.test.dot(matrix))
        return projection_trains, projection_tests

    def run(self):
        projections = self.projection()
        labels = (self.train_label, self.test_label)
        onenn(projections, labels, self.name, "PCA")

    def onenn(self):
        """计算1-NN,二分类"""
        print("====算法：PCA ==== \n%s文件测试结果：" % self.name)
        projection_trains, projection_tests = self.projection()

        for i in range(3):

            row_test, _ = projection_tests[i].shape
            row_train, _ = projection_trains[i].shape
            count = 0
            for row_index in range(row_test):
                dist_row = []
                for rowtrain in range(row_train):
                    dist_row.append(np.linalg.norm(projection_tests[i][row_index]-projection_trains[i][rowtrain]))

                result = self.train_label[dist_row.index(min(dist_row))]

                if result == self.test_label[row_index]:
                    count += 1

            if i == 0:
                print("k=10,正确率：{0}/{1}={2}%".format(count, row_test, 1.0*count/row_test*100))

            if i == 1:
                print("k=20,正确率：{0}/{1}={2}%".format(count, row_test, 1.0*count/row_test*100))

            if i == 2:
                print("k=30,正确率：{0}/{1}={2}%".format(count, row_test, 1.0*count/row_test*100))

    def onenn_use_knn(self):
        """使用knn模块计算1-nn，二分类"""
        print("====算法：PCA ==== \n%s文件测试结果：" % self.name)
        projection_trains, projection_tests = self.projection()
        for i in range(3):
            labels = knn(projection_trains[i], self.train_label, projection_tests[i], k=1)
            count = 0
            for index, label in enumerate(labels):
                count = count + 1 if label == self.test_label[index] else count

            if i == 0:
                print("k=10,正确率：{0}/{1}={2}%".format(count, len(labels), 1.0*count/len(labels)*100))

            if i == 1:
                print("k=20,正确率：{0}/{1}={2}%".format(count, len(labels), 1.0*count/len(labels)*100))

            if i == 2:
                print("k=30,正确率：{0}/{1}={2}%".format(count, len(labels), 1.0*count/len(labels)*100))
