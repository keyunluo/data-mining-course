#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午9:58
# @Author  : 骆克云
# @File    : projectutil.py
# @Software: PyCharm

import os
from collections import OrderedDict
import pandas as pd


### 作业1 工具方法

def get_corpus_filepaths():
    """
    获取语料库的类别名称及文件列表,并且排过序
    :return: 类名-文件路径字典
    """

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

    # 排序：使用类别前面的数字

    return OrderedDict(sorted(corpusfiles.items(), key=lambda x: int(x[0].split(".")[0])))


def get_corpus_filenames():
    """
    获得语料库中类名-文件名字典
    :return: 类名-文件名字典
    """
    corpusfiles = get_corpus_filepaths()
    for corpusfile in corpusfiles:
        files = corpusfiles.get(corpusfile)
        corpusfiles[corpusfile] = [file.split(os.sep)[-1] for file in files]

    return corpusfiles


def get_corpus_classfiles(classname):
    """
    通过类名获得该类下的所有文件名
    :param classname:类名
    :return: 文件列表
    """
    return sorted(corpus_filenames.get(classname, None))


def get_full_paths():
    """
    文件名：文件路径映射
    :return: 文件名-文件路径字典
    """
    full_path = {}
    for _, paths in corpus_filepaths.items():
        for path in paths:
            full_path[path.split(os.sep)[-1]] = path
    return full_path


def get_full_path(file):
    """
    由文件名获得文件路径
    :param file: 文件名
    :return: 文件路径
    """
    return get_full_paths().get(file)


def get_stopwords_file():
    """
    返回停用词文件路径
    :return: 停用词文件路径
    """
    return project_dir+os.sep+"data"+os.sep+"StopWords"+os.sep+"english"


project_dir = os.path.dirname(os.path.abspath(__file__)) # 项目路径
corpus_filepaths = get_corpus_filepaths() # 字典：类名-文件路径
corpus_filenames = get_corpus_filenames() #字典：类名-文件名


### 作业2 工具方法

def get_dataset_dir():
    """获得数据文件所在的文件夹"""
    return project_dir + os.sep + "data" + os.sep + "BinaryDatasets" + os.sep


def get_reduction_dataset(name="sonar", types="train"):
    """
    获取数据集
    :param name: 数据名称：sonar/splice
    :param types:数据类型：train/test
    :return: 数据，标签
    """
    file_train = get_dataset_dir() + name + "-train.txt"
    file_test = get_dataset_dir() + name + "-test.txt"
    file = file_train if types == "train" else file_test

    data = []
    label = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split(",")
            data.append(line[:-1])
            label.append(line[-1].strip("\n"))

    return data, label

### 作业3 获取数据

def get_handwritten_digits(filename):
    file_path = project_dir + os.sep + "data/Clustering/" + filename + ".txt"
    handwritten_digits = pd.read_csv(file_path, header=None)
    label_column = handwritten_digits.columns[-1]
    label = handwritten_digits[label_column]
    handwritten_digits = handwritten_digits.drop(label_column, axis=1)
    return handwritten_digits, label

## 作业4 获取数据


def get_sgd_data(name="dataset1-a9a", types="train"):
    """
    获取数据集
    :param name:数据名称：dataset1-a9a/covtype
    :param type:数据类型：train/test
    :return:数据，标签
    """
    file_train = project_dir + os.sep + "data/SGD/" + name + "-training.txt"
    file_test = project_dir + os.sep + "data/SGD/" + name + "-testing.txt"
    file = file_train if types == "train" else file_test

    data = []
    label = []

    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.split(",")
            data.append(line[:-1])
            label.append(line[-1].strip("\n"))

    return data, label

