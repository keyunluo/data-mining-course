#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-3 下午6:43
# @Author  : 骆克云
# @File    : ReductionTest.py
# @Software: PyCharm

from assignment2.DimReductionwithSklearn import Reduction


def test_reduction(name="sonar"):
    reduction = Reduction(name)
    reduction.pca_solver()
    reduction.svd_solver()
    reduction.isomap_solver(k_nn=4)
    reduction.isomap_solver(k_nn=6)
    reduction.isomap_solver(k_nn=8)
    reduction.isomap_solver(k_nn=10)

if __name__ == '__main__':
    for file in ["sonar", "splice"]:
        test_reduction(file)
