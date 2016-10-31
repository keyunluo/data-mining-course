#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-26 下午4:53
# @Author  : 骆克云
# @File    : kmedoids.py
# @Software: PyCharm

from projectutil import get_handwritten_digits
from assignment3.score import score
from scipy.spatial.distance import cdist
from multiprocessing import Pool
import random
import pandas as pd
import numpy as np


class KMedoids:
    def __init__(self, filename, repeat=0, max_iterator=1000, alg="K-Medoids"):
        handwritten_digits, true_label = get_handwritten_digits(filename)
        self.filename = filename
        self.true_label = true_label
        self.c = len(true_label.unique())
        self.data = handwritten_digits
        self.max_iterator = max_iterator
        self.label = true_label.unique()
        self.distance = cdist(self.data, self.data, metric="cityblock")
        self.repeat = repeat
        self.alg = alg

    def reinit(self, data):
        self.data = pd.DataFrame(data)
        self.distance = cdist(self.data, self.data, metric="cityblock")

    def init_centers(self, thread=0):
        """初始化c个样本作为中心点"""
        center = self.data.sample(n=self.c, random_state=np.random.RandomState())
        labels = pd.Series(index=self.data.index)
        labels[center.index] = self.label
        centers = pd.Series(data=center.index, index=self.label)
        return centers, labels

    def assign_label(self, centers, labels):
        """指派新的标签"""
        center_indexes = centers.values.tolist()
        for point_index in labels.index:

            index = center_indexes[np.argmin(self.distance[point_index][center_indexes])]
            '''
            short_dis = np.inf
            index = 0
            for center_index in center_indexes:
                dist = self.distance[point_index][center_index] #distance(points.loc[point_index].values, points.loc[center_index].values)
                if dist < short_dis:
                    short_dis = dist
                    index = center_index
            '''
            labels[point_index] = labels[index]
        return labels

    def update_medoids(self, centers, labels):
        """更新中心点"""
        for label in self.label:
            # 得到聚类中的点
            cluster_index = labels[labels == label].index
            # 计算中心点
            center = pd.Series(index=cluster_index)
            for point_index in cluster_index:
                """
                dist = 0
                for other_point_index in cluster_index:
                    dist += self.distance[point_index][other_point_index]  # distance(points.loc[point_index].values, points.loc[other_point_index].values)
                """
                center[point_index] = sum(self.distance[point_index][cluster_index])
            centers[label] = center.argmin()

        return centers

    def cost(self, centers, labels, label):
        """类内代价"""
        return sum(self.distance[centers[label]][point_index] for point_index in labels[labels == label].index)

    def manhadun_cost(self, centers, labels):
        """聚类代价"""
        all_cost = 0
        for label in self.label:
            all_cost += sum(self.distance[centers[label]][point_index] for point_index in labels[labels == label].index)
        return all_cost

    def exchange_value(self, X_Old, X_New, centers, labels):
        """交换值，替换中心点"""
        XOld_label = labels[X_Old]
        labels[X_Old] = labels[X_New]
        labels[X_New] = XOld_label
        centers[XOld_label] = X_New
        return centers

    def exchange_all(self, centers, labels):
        """交换所有值，替换中心点"""
        for label in self.label:
            for point_index in labels[labels == label].index: # labels.index:
                old_cost = self.cost(centers, labels, label) # self.manhadun_cost(centers, labels)
                old_center = centers[label]
                centers = self.exchange_value(old_center, point_index,centers, labels)
                # labels = self.assign_label(centers, labels)
                new_center = centers[label]
                new_cost = self.cost(centers, labels, label) # self.manhadun_cost(centers, labels)
                # 交换后结果变差，还原
                if new_cost > old_cost:
                    centers = self.exchange_value(new_center, old_center, centers, labels)
        labels = self.assign_label(centers, labels)
        return self.manhadun_cost(centers, labels)

    def exchange_random(self, centers, labels, r_ration=0.4):
        """随机交换部分值，替换中心点"""
        for label in self.label:
            indexes = labels[labels == label].index
            selected_indexes = random.sample(indexes.tolist(), int(r_ration*len(indexes)))
            for point_index in selected_indexes:  # labels.index:
                old_cost = self.cost(centers, labels, label) # self.manhadun_cost(centers, labels)
                old_center = centers[label]
                centers = self.exchange_value(old_center, point_index, centers, labels)
                #labels = self.assign_label(centers, labels)
                new_center = centers[label]
                new_cost = self.cost(centers, labels, label) # self.manhadun_cost(centers, labels)
                # 交换后结果变差，还原
                if new_cost > old_cost:
                    centers = self.exchange_value(new_center, old_center, centers, labels)
        labels = self.assign_label(centers, labels)
        return self.manhadun_cost(centers, labels)

    def exchange(self, centers, labels, alg="random"):
        if alg == "random":
            return self.exchange_random(centers, labels)
        return self.exchange_all(centers, labels)

    def kmedoids_alg(self, centers, labels, thread, alg="random"):
        count = 0
        self.assign_label(centers, labels)
        while True:
            old_cost = self.manhadun_cost(centers, labels)
            new_cost = self.exchange(centers, labels, alg)
            #print(old_cost, new_cost)
            if new_cost >= old_cost or count > self.max_iterator:
                purity_value, gini_value = score(labels, self.true_label)
                print("file: %s, 算法: %s, 第%d次运行结果:" % (self.filename, self.alg, thread + 1))
                print("result: purity:%f, gini:%f, center_points:%s" % (purity_value, gini_value, centers.values))
                break
            count += 1

    def medoids(self, centers, labels, thread):
        # print("init:", centers.values)
        count = 0
        new_labels = self.assign_label(centers, labels)
        while True:
            old_centers = centers.copy()
            new_centers = self.update_medoids(centers, new_labels)
            new_labels = self.assign_label(centers, labels)
            if old_centers.equals(new_centers)or count > self.max_iterator:
                purity_value, gini_value = score(new_labels, self.true_label)
                print("file: %s, 算法: %s, 第%d次运行结果:" % (self.filename, self.alg, thread+1))
                print("result: purity:%f, gini:%f, center_points:%s" % (purity_value, gini_value, centers.values))
                break
            count += 1

    def main_func(self, thread=0):
        centers, labels = self.init_centers(thread)
        #self.medoids(centers, labels, thread)
        self.kmedoids_alg(centers, labels, thread) # , alg="all"

    def multi_core(self):
        pool = Pool()
        pool.map(self.main_func, range(self.repeat))
        pool.close()
        pool.join()

    def run(self):
        if self.repeat == 0:
            self.main_func()
        else:
            self.multi_core()

