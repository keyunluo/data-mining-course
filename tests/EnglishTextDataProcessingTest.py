#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午7:22
# @Author  : 骆克云
# @File    : EnglishTextDataProcessingTest.py
# @Software: PyCharm


from projectutil import project_dir
from assignment1.EnglishTextDataProcessing import preprocess
from assignment1.EnglishTextDataProcessing import dataset


def test_preprocess():
    tokenizing = preprocess.token(project_dir + "/data/ICML/1. Active Learning/Diagnosis determination.txt")
    stemming = preprocess.stemming(tokenizing)
    stopwords = preprocess.filter_stopwords(stemming)
    print(stopwords)

def test_dataset():
    dataset.make_dataset()

if __name__ == '__main__':
    #test_preprocess()
    test_dataset()

