#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午4:29
# @Author  : 骆克云
# @File    : tfidf.py
# @Software: PyCharm

from collections import defaultdict
from math import log
from assignment1.EnglishTextDataProcessing.preprocess import get_words
from projectutil import get_full_path
from projectutil import corpus_filenames
from projectutil import get_corpus_classfiles


class TFIDF:
    """
    tf-idf算法
    """

    def __init__(self):
        self.numdoc = 0
        self.document = {}
        self.wordfile = defaultdict(set)
        self.cangeneratedataset = False

    def add_doc(self, file):
        """添加文件，指定文件名"""
        words = get_words(get_full_path(file))  #文本预处理
        self.numdoc += 1
        worddict = {}

        for word in words:
            worddict[word] = worddict.get(word, 0.0) + 1.0
            self.wordfile[word].add(file)

        # 归一化
        length = float(len(words))

        for word in worddict:
            worddict[word] = worddict.get(word)/length

        self.document[file] = worddict
        self.cangeneratedataset = True

    def add_doc_class(self, classname):
        """添加文件，指定类别"""
        classfile = get_corpus_classfiles(classname)
        for file in classfile:
            self.add_doc(file)

    def add_doc_all(self):
        """添加所有语料库中的文件"""
        for _, files in corpus_filenames.items():
            for file in files:
                self.add_doc(file)

    def tf_idf(self, file):
        """TF-IDF算法:单个文件"""
        worddict = self.document[file]
        wordfile = self.wordfile
        numdoc = self.numdoc
        tfidfdict = {}
        for word in worddict:
            tf = worddict.get(word)
            idf = log(numdoc/len(wordfile.get(word)))
            tfidfdict[word] = tf*idf

        return tfidfdict

    def tfidf_class(self, classname):
        """TF-IDF算法:类名"""
        tfidfdictclass = {}
        classfiles = get_corpus_classfiles(classname)
        for classfile in classfiles:
            tfidfdictclass[classfile] = self.tf_idf(classfile)

        return tfidfdictclass

    def tfidf_all(self):
        """TF-IDF算法:所有文件"""
        tfidfdictall = {}
        for classname in corpus_filenames:
            tfidfdictall[classname] = self.tfidf_class(classname)

        return tfidfdictall

    def generate_dataset(self, file):
        """产生数据集"""
        if not self.cangeneratedataset:
            print("还没有分析文档，无法产生数据集")
            return False

        docs = [doc for doc in self.document]
        docs.sort()
        print(docs)
        print("文档数为：%d"%len(docs))
        words = [word for word in self.wordfile]
        print(len(words))

    def generate_dataset_class(self, classname):
        classfiles = get_corpus_classfiles(classname)
        for classfile in classfiles:
            self.generate_dataset(classfile)

    def generate_dataset_all(self):
        for classname in corpus_filenames:
            self.generate_dataset_class(classname)
