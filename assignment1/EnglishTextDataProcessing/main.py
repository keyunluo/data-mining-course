#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午4:31
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm


from assignment1.EnglishTextDataProcessing import dataset
import time

def main():
    print("====开始进行处理=====")
    time_start = time.time()
    dataset.make_dataset()
    time_end = time.time()
    print("共运行时间：%d秒" % int(time_end - time_start))
    print("======结束处理=======")
