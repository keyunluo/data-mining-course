#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-10 下午3:56
# @Author  : 骆克云
# @File    : sgdwithsklearn.py
# @Software: PyCharm

from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from projectutil import get_sgd_data
import numpy as np


class SGDWithSklearn:
    def __init__(self, filename, sc_alg="mmsc"):
        data_train, label_train = get_sgd_data(filename, "train")
        data_test, label_test = get_sgd_data(filename, "test")
        sc = preprocessing.StandardScaler() if sc_alg == "stdsc" else preprocessing.MinMaxScaler()
        self.train = sc.fit_transform(np.array(data_train).astype(float))
        #self.train = np.array(data_train).astype(float)
        self.train_label = np.array(label_train).astype(int)

        self.test = sc.transform(np.array(data_test).astype(float))
        #self.test = np.array(data_test).astype(float)
        self.test_label = np.array(label_test).astype(int)
        self.filename = filename

    def score(self, prob):
        return -1 if prob < 0 else 1

    def sgd(self):
        clf = SGDClassifier(loss="log", penalty="l1", shuffle=True)
        clf.fit(self.train, self.train_label)
        '''
        prob_train = clf.predict_proba(self.train)
        prob_test = clf.predict_proba(self.test)
        score_train = sum(1 if self.score(prob) == self.train_label[index] else 0 for index, prob in enumerate(prob_train)) / len(prob_train)
        score_test = sum(self.score(prob) == self.test_label[index] for index, prob in enumerate(prob_test)) / len(prob_test)
        '''
        predict_train = clf.predict(self.train)
        predict_test = clf.predict(self.test)
        score_train = sum(
            1 if predict_train[index] == self.train_label[index] else 0 for index in range(len(self.train))) / len(
            self.train)
        score_test = sum(predict_test[index] == self.test_label[index] for index in range(len(self.test))) / len(
            self.test)

        print("文件：%s，" % self.filename, "算法：sklearn-sgd-logloss，", "训练集正确率: %.4f，" % score_train, "测试集正确率: %.4f" % score_test)

    def rr(self):
        #clf = SGDClassifier(loss="squared_loss", penalty="l2", shuffle=True, alpha=0.01)
        clf = SGDRegressor(loss="squared_loss", penalty="l2", shuffle=True)
        clf.fit(self.train, self.train_label)
        predict_train = clf.predict(self.train)
        predict_test = clf.predict(self.test)
        score_train = sum(self.score(predict_train[index]) == self.train_label[index] for index in range(len(self.train))) / len(
                self.train)
        score_test = sum(self.score(predict_test[index]) == self.test_label[index] for index in range(len(self.test))) / len(
            self.test)
        print("文件：%s，" % self.filename, "算法：sklearn-sgd-ridgeregression，", "训练集正确率: %.4f，" % score_train,
              "测试集正确率: %.4f" % score_test)

    def run(self):
        self.sgd()
        self.rr()
