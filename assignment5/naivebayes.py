#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-17 下午5:06
# @Author  : 骆克云
# @File    : naivebayes.py
# @Software: PyCharm


from projectutil import get_ensemble_data
from .kfolddataset import train_test_split
import numpy as np
from collections import defaultdict


class NaiveBayes:
    """
    朴素贝叶斯分类器
    """
    def __init__(self):
        self.train = None
        self.train_label = None
        self.label = None
        self.test = None
        self.test_label = None
        self.all_data = None
        self.feature_types = None
        self.gauss_args = defaultdict(dict)
        self.generating_model = {}

    def gaussian_func(self, feature, label, x):
        """高斯函数"""
        mean, var = self.gauss_args[feature][label]
        return 1.0 / (np.sqrt(2*np.pi*var)) * np.exp(-(x - mean)*(x - mean) / (2*var))

    def train_data(self):
        """训练"""
        length = len(self.train)

        # 计算先验概率和条件概率
        for label in self.label:
            train_model = []
            indexes = np.argwhere(self.train_label == label).flatten()
            data = self.train[indexes]
            train_lenth = len(indexes)
            prior_prob = (1.0 * train_lenth) / length
            train_model.append(prior_prob)

            for index, feature_type in enumerate(self.feature_types):
                # 离散变量, 计算条件概率
                if feature_type == 1:
                    unique_dict = {}
                    unique_value = np.unique(self.all_data[:, index])
                    for val in unique_value:
                        unique_dict[val] = (len(np.where(data[:, index] == val)[0]) + 1.0) / (train_lenth + len(self.label))
                    train_model.append(unique_dict)
                # 连续特征离散化：高斯模型
                else:
                    data_train = self.train[indexes, index]
                    mean = data_train.mean()
                    var = data_train.var()
                    self.gauss_args[index][label] = [mean, var]
            self.generating_model[label] = train_model

    def calculate_prob(self, data, label):
        """计算测试集的概率"""
        model = self.generating_model[label]
        feature_index = 0
        test_prob = np.full(len(data), model[feature_index], dtype=np.float)
        for index, feature_type in enumerate(self.feature_types):
            # 数值连续特征， 计算概率值
            if feature_type == 0:
                test_prob *= [self.gaussian_func(index, label, x) for x in data[:, index]]
            # 离散特征， 直接求概率
            else:
                feature_index += 1
                condition = model[feature_index]
                test_prob *= [condition[x] for x in data[:, index]]
        return test_prob

    def train_predict(self):
        """训练值预测"""
        pred = np.array([self.calculate_prob(self.train, label) for label in self.label])
        pred = pred.argmax(axis=0)
        pred[pred == 0] = self.label[0]
        pred[pred == 1] = self.label[1]
        return pred

    def predict(self):
        """预测测试集"""
        pred = np.array([self.calculate_prob(self.test, label) for label in self.label])
        test_pred = pred.argmax(axis=0)
        test_pred[test_pred == 0] = self.label[0]
        test_pred[test_pred == 1] = self.label[1]

        return test_pred

    def score(self, filename, pred, alg="Naive Bayes"):
        """准确率评估"""
        length = len(pred)
        correct = len(np.where(pred == self.test_label)[0])
        print("================================================================")
        print("文件:%s" % filename, "算法:%s" % alg, "正确率: %.4f" % (1.0 * correct / length))
        print("================================================================")

    def fit(self, all_data, label, feature_types, train_index, test_index):
        self.all_data = all_data
        self.label = np.unique(label)
        self.feature_types = feature_types
        self.train = self.all_data[train_index]
        self.train_label = label[train_index]
        self.test = self.all_data[test_index]
        self.test_label = label[test_index]
        self.gauss_args = defaultdict(dict)
        self.generating_model = {}

    def run(self, filename):
        data, label, feature_types = get_ensemble_data(filename)
        data = np.array(data).astype(float)
        self.all_data = data
        label = np.array(label).astype(float)
        label = label.astype(int)
        self.feature_types = np.array(feature_types).astype(int)
        train, test = train_test_split(data)
        self.fit(self.all_data, label, self.feature_types, train, test)
        self.train_data()
        test_pred = self.predict()
        self.score(filename, test_pred)
