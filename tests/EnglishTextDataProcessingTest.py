#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午7:22
# @Author  : 骆克云
# @File    : EnglishTextDataProcessingTest.py
# @Software: PyCharm


from projectfile import project_dir
from assignment1.EnglishTextDataProcessing import preprocess


def test_preprocess():
    tokenizing = preprocess.token(project_dir + "/data/ICML/1. Active Learning/Diagnosis determination.txt")
    stemming = preprocess.stemming(tokenizing)
    stopwords = preprocess.filter_stopwords(stemming)
    print(stopwords)

if __name__ == '__main__':
    test_preprocess()
