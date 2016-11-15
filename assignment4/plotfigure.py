#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-10 下午4:17
# @Author  : 骆克云
# @File    : plotfigure.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['Kaiti'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号


def plot_figure(train_error, test_error, filename, algorithm, scaler):
    length = len(train_error)
    x = np.linspace(0, length*scaler, num=length, endpoint=True)
    plt.figure(figsize=(18, 8))
    plt.plot(x, train_error, 'b-', label='train')
    plt.plot(x, test_error, 'rx-', label='test')
    plt.axhline(y=min(train_error), linewidth=2, color='g', label='训练集最小错误率：%.4f' % min(train_error))
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('错误率')
    plt.title('文件：%s, 算法: %s, training/testing错误率' % (filename, algorithm))
    plt.legend()
    save_name = 'assignment4/figure/' + '%s_%s_错误率' % (filename, algorithm)
    plt.savefig(save_name)
    #plt.show()


def plot_objective_function(objective_function, filename, algorithm, function, scaler):
    length = len(objective_function)
    x = np.linspace(0, length * scaler, num=length, endpoint=True)
    plt.figure(figsize=(14, 8))
    plt.plot(x, objective_function, 'b-', label='目标函数')
    plt.axhline(y=min(objective_function), linewidth=2, color='g', label='目标函数最小值：%.4f' % min(objective_function))
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('目标函数:%s' % function)
    plt.title('文件：%s, 算法: %s, 目标函数:%s' % (filename, algorithm, function))
    plt.legend()
    save_name = 'assignment4/figure/' + '%s_%s_目标函数' % (filename, algorithm)
    plt.savefig(save_name)
    #plt.show()
