#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-18 上午9:58
# @Author  : 骆克云
# @File    : projectutil.py
# @Software: PyCharm

import os


def get_corpus_filepaths():
    """
    获取语料库的类别名称及文件列表
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

    return corpusfiles


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
    return corpus_filenames.get(classname, None)


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
