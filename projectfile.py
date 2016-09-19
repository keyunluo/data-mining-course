#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午9:58
# @Author  : 骆克云
# @File    : projectfile.py
# @Software: PyCharm

import os

def get_corpus_files():
    '''获取语料库的类别名称及文件列表'''

    corpusfiles ={}
    filedir = project_dir+os.sep+"data"+os.sep+"ICML"+os.sep
    for root, dirs, files in os.walk(filedir):
        key = root.split(os.sep)[-1]
        value = []
        for file in files:
            if file.endswith(".txt"):
                value.append(os.path.join(root, file))

        if len(value) > 0:
            corpusfiles[key] = value

    return corpusfiles

def get_stopwords_file():
    '''返回停用词文件路径'''
    return project_dir+"/data/StopWords/english"

project_dir = os.path.dirname(os.path.abspath(__file__))
corpus_files = get_corpus_files()
