#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-17 下午5:06
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm

from assignment5.adaboost import AdaBoost


def run():
    for file in ["breast-cancer", "german"]: # "breast-cancer", , "german"
        adaboost = AdaBoost()
        adaboost.run(file)
