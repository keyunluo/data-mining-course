#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-17 下午5:06
# @Author  : 骆克云
# @File    : adaboost.py
# @Software: PyCharm

from .kfolddataset import k_fold_train_test_split, train_test_split
from .naivebayes import NaiveBayes
from projectutil import get_ensemble_data
import numpy as np


class AdaBoost:
    """提升方法"""
    def __init__(self, max_iter=5, k_fold=10):
        self.classifier = NaiveBayes()
        self.max_iter = max_iter
        self.k_fold = k_fold
        self.data = None
        self.label = None
        self.feature_types = None
        self.filename = None
        
    def sample(self, indexes, weights):
        # 带权随机取样， 产生新的训练集
        return [np.random.choice(indexes, p=weights) for _ in indexes]

    def boost(self, train_index, test_index):
        # 训练集测试集
        train_label = self.label[train_index]
        test = self.data[test_index]
        test_label = self.label[test_index]

        length = len(train_index)
        # 训练集索引
        indexes = np.arange(length)
        # 权值, 初始均为1/N
        D = np.full(length, 1.0 / length, dtype=np.float)
        base_classifier = self.classifier

        # 基本分类器的线性组合
        fx = np.zeros(len(test))
        # m次迭代
        for m in np.arange(self.max_iter):
            # 基本分类器初始化
            base_classifier.fit(self.data, self.label, self.feature_types, train_index, test_index)
            base_classifier.train_data()
            # 训练数据集的结果
            pred = base_classifier.train_predict()
            # 测试数据集的结果
            pred_test = base_classifier.predict()

            # NaiveBayes 分类器
            #if m == 0:
            #    correct = len(np.where(pred_test == test_label)[0])
            #    print("文件:%s" % self.filename, "算法:%s" % "Naive Bayes", "正确率: %.4f" %
            #          (1.0 * correct / len(test_label)), end=';')

            # 错误预测的索引项
            predict_wrong = np.argwhere(pred != train_label).flatten()
            # 错误率
            error = np.sum(D[predict_wrong])
            # 系数
            alpha = 0.5 * np.log((1 - error) / error)
            # 规范化因子 
            Z = sum([D[i]*np.exp(-alpha*train_label[i]*pred[i]) for i in range(length)])
            for i in indexes:
                D[i] = D[i]*np.exp(-alpha*train_label[i]*pred[i]) / Z

            fx += alpha*pred_test
            train_index = self.sample(train_index, D)
            #train_index = self.sample(indexes, D)
            
        # 最终分类器
        Gx = np.sign(fx)
        correct = len(np.where(Gx == test_label)[0])
        score = (1.0 * correct) / len(test)
        print("文件:%s" % self.filename, "  算法:%s" % "AdaBoost", "正确率: %.4f" % score)
        return score

    def cross_validation(self, train_test_data):
        score = []
        for i in range(self.k_fold):
            train_index = train_test_data[0][i]
            test_index = train_test_data[1][i]
            score.append(self.boost(train_index, test_index))
        return score

    def run(self, filename):
        data, label, feature_types = get_ensemble_data(filename)
        self.filename = filename
        self.data = np.array(data).astype(float)
        label = np.array(label).astype(float)
        label[label == 0] = -1
        self.label = label
        self.feature_types = np.array(feature_types).astype(int)
        train_test_data = k_fold_train_test_split(self.k_fold, self.data)
        score = self.cross_validation(train_test_data)
        print("==========================================================")
        print("文件:%s" % self.filename,"算法：%s" % "AdaBoost", "测试集准确率 平均值：%.4f" % np.mean(score), "标准差: %.4f" % np.std(score))
        print("==========================================================")
