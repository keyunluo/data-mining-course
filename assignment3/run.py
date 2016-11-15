#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-10-26 下午5:05
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm


from assignment3.kmedoids import KMedoids
from assignment3.spectralclustering import Spectral
from multiprocessing import Pool


def multicore(files):
    pool = Pool()
    pool.map(parallelize_run, files)
    pool.close()
    pool.join()

def parallelize_run(file):
    kmedoids = KMedoids(file, repeat=10)
    kmedoids.run()
    spectral = Spectral(file, knn=3)
    spectral.run()
    spectral = Spectral(file, knn=6)
    spectral.run()
    spectral = Spectral(file, knn=9)
    spectral.run()

def run():
    for file in ["german","mnist"]:  # , "mnist" "german",
        kmedoids = KMedoids(file, repeat=10)
        kmedoids.run()
        spectral = Spectral(file,knn=3)
        spectral.run()
        spectral = Spectral(file,knn=6)
        spectral.run()
        spectral = Spectral(file,knn=9)
        spectral.run()



if __name__ == "__main__":
    run()

