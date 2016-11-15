#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-10 下午4:00
# @Author  : 骆克云
# @File    : logisticregression.py
# @Software: PyCharm

import numpy as np
from projectutil import get_sgd_data
from .plotfigure import plot_figure
from .plotfigure import plot_objective_function

class LRWithSGD:
    def __init__(self, filename, max_iter=5, lambda_value=0.001):
        data_train, label_train = get_sgd_data(filename, "train")
        data_test, label_test = get_sgd_data(filename, "test")

        raw_data_train = np.array(data_train).astype(float)
        raw_data_test = np.array(data_test).astype(float)

        #min_val, denominator = self.fit_minmax_scale(raw_data_train)
        #self.train = self.transform(raw_data_train, min_val, denominator)
        mean, std_variance = self.all_scale(raw_data_train)
        self.train = self.scale(raw_data_train, mean, std_variance)
        #self.train = raw_data_train
        m, _ = self.train.shape
        self.train = np.c_[np.ones(m), self.train]
        self.train_label = np.array(label_train).astype(int)

        #self.test = self.transform(raw_data_test, min_val, denominator)
        self.test = self.scale(raw_data_test, mean, std_variance)
        #self.test = raw_data_test
        m, _ = self.test.shape
        self.test = np.c_[np.ones(m), self.test]
        self.test_label = np.array(label_test).astype(int)

        self.filename = filename
        self.max_iter = max_iter
        self.lambda_value = lambda_value
        self.scaler = int(len(self.train) * self.max_iter / 100)

    def all_scale(self, data):
        mean = data.mean(axis=0).T
        std_variance = data.std(axis=0).T
        std_variance[std_variance == 0] = 1
        return mean, std_variance

    def scale(self, data, mean, std_variance):
        data = (data - mean) / std_variance
        return data

    def fit_minmax_scale(self, data):
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        denominator = max_val -min_val
        denominator[denominator == 0] = 1
        return min_val, denominator

    def transform(self, data, min_val, denominator):
        return (data - min_val) / denominator

    def sigmod(self, x):
        return 1.0 / (1 + np.exp(-x))

    def decent(self, index, beta, k):
        return (self.sigmod(self.train_label[index]*beta.dot(self.train[index].T)) - 1)*self.train_label[index]*self.train[index][k]

    def decent_vector(self, index, beta):
        return (self.sigmod(self.train_label[index]*beta.dot(self.train[index].T)) - 1)*self.train_label[index]*self.train[index]

    def sign(self, w):
        if w > 0:
            return 1
        elif w < 0:
            return -1
        return 0

    def gradient(self):
        m, n = self.train.shape
        beta = np.ones(n)
        beta[0] = 0
        history_beta = []
        indexes = np.array(range(m))

        for j in range(self.max_iter):
            if j % 5000 == 0:
                np.random.shuffle(indexes)
            alpha = 1.0 / (1.0 + j/200) + 0.001

            index = np.random.choice(indexes)
            history_beta.append(beta)
            beta = beta - alpha*(self.decent_vector(index, beta) + self.lambda_value*self.sign_vector(beta))
        print("文件：%s，" % self.filename, "算法：LRWithSGD-Logloss，",
              "训练集准确率：", self.score_datasets(beta, self.train, "train"),
              "测试集准确率：", self.score_datasets(beta, self.test, "test"))
        return beta, history_beta

    def sign_vector(self, w):
        sign_value = np.copy(w)
        sign_value[sign_value > 0] = 1
        sign_value[sign_value < 0] = -1
        return sign_value

    def sgd(self):
        m, n = self.train.shape
        beta = np.ones(n)
        beta[0] = 0
        history_beta = []
        indexes = np.array(range(m))
        alpha0 = 8.0 / (len(self.train))
        for j in range(self.max_iter):
            np.random.shuffle(indexes)
            for i in indexes:
                alpha = 1.0 / (4.0 + i + j) + alpha0
                history_beta.append(beta.copy())
                beta = beta - alpha * (self.decent_vector(i, beta) + self.lambda_value * self.sign_vector(beta))

        return history_beta

    def plot_score(self, history_beta):
        history_score = [[] for i in [0, 1]]
        best_index = 0
        min_train_err = 1
        for index, beta in enumerate(history_beta):
            if index % (self.scaler + 1) == 0:
                train_err = 1 - self.score_datasets(beta, self.train, "train")
                test_err = 1 - self.score_datasets(beta, self.test, "test")
                if train_err <= min_train_err:
                    min_train_err = train_err
                    best_index = index
                history_score[0].append(float('%.4f' % train_err))
                history_score[1].append(float('%.4f' % test_err))
        print("==================================================")
        beta = history_beta[best_index]
        print("文件：%s，" % self.filename, "算法：SGD-LogLossRegression，",
              "训练集准确率：", self.score_datasets(beta, self.train, "train"),
              "测试集准确率：", self.score_datasets(beta, self.test, "test"))
        print("训练集：\n", [score for index, score in enumerate(history_score[0]) if index % 10 == 0 or index == len(history_score[0])])
        print("测试集：\n", [score for index, score in enumerate(history_score[1]) if index % 10 == 0 or index == len(history_score[1])])
        print("==================================================")
        plot_figure(history_score[0], history_score[1], self.filename, "SGD-LogLossRegression-Error-Rate", self.scaler)

    def plot_objective_function(self, history_beta):
        history_obj = []
        for index, beta in enumerate(history_beta):
            if index % (self.scaler+1) == 0:
                obj_val = self.objective_function(beta, self.train, self.train_label, self.lambda_value)
                history_obj.append(float('%.4f' % obj_val))
        print("==================================================")
        print("文件：%s，" % self.filename, "算法：SGD-LogLossRegression，", "目标函数最小值：%s" % min(history_obj))
        print("目标函数：\n", [value for index, value in enumerate(history_obj) if index % 10 == 0 or index == len(history_obj)])
        print("==================================================")
        function = "$\min_\\beta \ \log(1 + e^{-y_i \\beta^T x_i})+ \lambda \Vert \\beta \Vert_1$"
        plot_objective_function(history_obj, self.filename, "SGD-LogLossRegression-Objective-Function", function, self.scaler)

    def score(self, prob):
        if prob >= 0.5:
            return 1
        return -1

    def score_datasets(self, beta, datas, types):
        label = self.train_label if types == "train" else self.test_label
        score = 1.0*sum(self.score(beta.dot(data.T)) == label[index] for index, data in enumerate(datas)) / len(datas)
        return float("%.4f" % score)

    def objective_function(self, beta, data, label, lambda_value):
        N = len(data)
        y = label
        x = data
        return 1.0 / N * sum(np.log(1 + np.exp(-y[i]*beta.dot(x[i].T))) for i in range(N)) + lambda_value*beta.sum()

    def run(self):
        #beta, history_beta = self.gradient()
        history_beta = self.sgd()
        self.plot_score(history_beta)
        self.plot_objective_function(history_beta)
