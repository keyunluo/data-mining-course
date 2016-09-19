#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午4:29
# @Author  : 骆克云
# @File    : dataset.py
# @Software: PyCharm

from assignment1.EnglishTextDataProcessing.tfidf import TFIDF

def make_dataset():
    """生成数据集"""
    tfidf = TFIDF()
    tfidf.add_doc_all()
    tfidfdictall = tfidf.tfidf_all()
    tfidf.generate_dataset_all()