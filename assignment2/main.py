#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-28 下午8:13
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm

from assignment2.pca import PCA
from assignment2.svd import SVD
from assignment2.isomap import ISOMAP


def run():

    for file in ["sonar","splice"]: # ,"splice"
        pca = PCA(file)
        pca.run()

        svd = SVD(file)
        svd.run()

        isomap = ISOMAP(file)
        isomap.run()

        isomap = ISOMAP(file, 6)
        isomap.run()

        isomap = ISOMAP(file,8)
        isomap.run()

        isomap = ISOMAP(file,10)
        isomap.run()




