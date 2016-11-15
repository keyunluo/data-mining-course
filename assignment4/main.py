#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-11-7 下午7:11
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm

from assignment4.logisticregression import LRWithSGD
from assignment4.ridgeregression import RRWithSGD
from assignment4.sgdwithsklearn import SGDWithSklearn

def run():

    for file in ["dataset1-a9a", "covtype"]: #, "covtype" "dataset1-a9a",
        lr = LRWithSGD(file)
        lr.run()
        rr = RRWithSGD(file)
        rr.run()

        sgd = SGDWithSklearn(file)
        sgd.run()

