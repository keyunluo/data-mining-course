#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-10 下午4:00
# @Author  : 骆克云
# @File    : ridgeregression.py
# @Software: PyCharm

import numpy as np
from projectutil import get_sgd_data
from .plotfigure import plot_figure
from .plotfigure import plot_objective_function


class RRWithSGD:
    def __init__(self, filename, max_iter=5, lambda_value=0.3):
        data_train, label_train = get_sgd_data(filename, "train")
        data_test, label_test = get_sgd_data(filename, "test")

        raw_data_train = np.array(data_train).astype(float)
        raw_data_test = np.array(data_test).astype(float)

        mean, std_variance = self.all_scale(raw_data_train)
        self.train = self.scale(raw_data_train, mean, std_variance)
        m, _ = self.train.shape
        self.train = np.c_[np.ones(m), self.train]
        self.train_label = np.array(label_train).astype(int)

        self.test = self.scale(raw_data_test, mean, std_variance)
        m, _ = self.test.shape
        self.test = np.c_[np.ones(m), self.test]
        self.test_label = np.array(label_test).astype(int)

        self.filename = filename
        self.max_iter = max_iter
        self.lambda_value = lambda_value
        self.scaler = int(len(self.train)*self.max_iter / 100)

    def all_scale(self, data):
        mean = data.mean(axis=0).T
        std_variance = data.std(axis=0).T
        std_variance[std_variance == 0] = 1
        return mean, std_variance

    def scale(self, data, mean, std_variance):
        data = (data - mean) / std_variance
        return data

    def score(self, data, beta):
        return -1 if beta.dot(data.T) < 0 else 1

    def score_datasets(self, beta, datas, types):
        label = self.train_label if types == "train" else self.test_label
        score = 1.0*sum(self.score(data, beta) == label[index] for index, data in enumerate(datas)) / len(datas)
        return float("%.4f" % score)

    def sgd(self):
        X = self.train
        y = self.train_label
        m, n = X.shape
        beta = np.ones(n)
        beta[0] = 0
        history_beta = [beta.copy()]
        indexes = np.array(range(m))
        #alpha0 = 1.0 / (len(self.train))
        count = 0
        for j in range(self.max_iter):
            np.random.shuffle(indexes)
            #alpha = 0.0001
            for i in indexes:
                #alpha = 1.0 / ((1.0 + j + i)*self.lambda_value)
                #alpha = 0.0001
                #alpha = 1.0 / (12*(1 + i)*(j+1)) + alpha0
                #alpha = 1.0 / (100 + i + j) + alpha0
                #alpha = 1.0 / (100.0 + i + j) + alpha0
                count += 1
                #alpha = 1.0 / (1 + count * (index + 1)) + alpha0
                #alpha = 1.0 / (12 * (1 + i) * (j + 1)) + 0.0001
                alpha = 0.0008 / np.sqrt(count) + 0.0001
                beta += alpha * (X[i]*(y[i] - beta.dot(X[i].T)) - self.lambda_value * beta)
                history_beta.append(beta.copy())

        return history_beta

    def plot_score(self, history_beta):
        history_score = [[] for i in [0, 1]]
        best_index = 0
        min_train_err = 1
        for index, beta in enumerate(history_beta):
            if index % (self.scaler+1) == 0:
                train_err = 1 - self.score_datasets(beta, self.train, "train")
                test_err = 1 - self.score_datasets(beta, self.test, "test")
                if train_err <= min_train_err:
                    min_train_err = train_err
                    best_index = index
                history_score[0].append(float('%.4f' % train_err))
                history_score[1].append(float('%.4f' % test_err))
        beta = history_beta[best_index]
        print("==================================================")
        print("文件：%s，" % self.filename, "算法：SGD-RidgeRegression，",
              "训练集准确率：", self.score_datasets(beta, self.train, "train"),
              "测试集准确率：", self.score_datasets(beta, self.test, "test"))
        print("训练集：\n", [score for index, score in enumerate(history_score[0]) if index % 10 == 0 or index == len(history_score[0])])
        print("测试集：\n", [score for index, score in enumerate(history_score[1]) if index % 10 == 0 or index == len(history_score[1])])
        print("==================================================")
        plot_figure(history_score[0], history_score[1], self.filename, "SGD-RidgeRegression-Error-Rate", self.scaler)

    def plot_objective_function(self, history_beta):
        history_obj = []
        for index, beta in enumerate(history_beta):
            if index % (self.scaler+1) == 0:
                obj_val = self.objective_function(beta, self.train, self.train_label, self.lambda_value)
                history_obj.append(float('%.4f' % obj_val))
        print("==================================================")
        print("文件：%s，" % self.filename, "算法：SGD-RidgeRegression，", "目标函数最小值：%s" % min(history_obj))
        print("目标函数：\n", [value for index, value in enumerate(history_obj) if index % 10 == 0 or index == len(history_obj)])
        print("==================================================")
        function = "$\min_\\beta \ (y_i - \\beta^Tx_i)^2 + \lambda \Vert \\beta \Vert^2_2$"
        plot_objective_function(history_obj, self.filename, "SGD-RidgeRegression-Objective-Function", function, self.scaler)

    def objective_function(self, beta, data, label, lambda_value):
        N = len(data)
        y = label
        x = data
        return 1.0 / N * sum((y[i] - beta.dot(x[i].T))*(y[i] - beta.dot(x[i].T)) for i in range(N)) + lambda_value*np.linalg.norm(beta, 2)*np.linalg.norm(beta, 2)

    def run(self):
        history_beta = self.sgd()
        self.plot_score(history_beta)
        self.plot_objective_function(history_beta)
