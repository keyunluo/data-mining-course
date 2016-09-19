#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午4:31
# @Author  : 骆克云
# @File    : main.py
# @Software: PyCharm

import os
from assignment1.EnglishTextDataProcessing import preprocess
from projectfile import project_dir
from projectfile import corpus_files

def process(directory = " "):
    '''
    载入预料库，指定文本目录
    :param directory: 语料库一级路径
    :return:
    '''

    if directory == " ":
        directory = project_dir+"/data/ICML"

    base_path = os.path.abspath(directory)

    for classes in os.listdir(directory):
        '''语料库二级路径'''
        class_path = os.path.join(base_path, classes)
        for item in os.listdir(class_path):
            item_path = os.path.join(class_path, item)
            if os.path.isfile(item_path) and item.endswith('.txt'):
                print(item)


def main():
    tokenizing = preprocess.token(project_dir + "/data/ICML/1. Active Learning/Diagnosis determination.txt")
    stemming = preprocess.stemming(tokenizing)
    stopwords = preprocess.filter_stopwords(stemming)
    print(stopwords)


