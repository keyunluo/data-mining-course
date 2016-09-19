#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 16-9-17 下午4:28
# @Author  : 骆克云
# @File    : preprocess.py
# @Software: PyCharm

'''分词,词干提取模块,去除停用词'''


import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet as wn
import string
import itertools
from projectfile import get_stopwords_file


def token(file):
    '''
    分词，去除乱码，统一转换成小写
    :param file: 文件路径
    :return: 单词列表
    '''

    pattern = r"""(?x)               # set flag to allow verbose regexps
              (?:[a-z]\.)+           # abbreviations, e.g. u.s.a.
              |\d+(?:\.\d+)?%?       # numbers, incl. currency and percentages
              |\w+(?:[-']\w+)*       # words with optional internal hyphens/apostrophe
              |\.\.\.                # ellipsis
              |(?:[.,;"'?():-_`])    # special characters with meanings
              """

    words = []

    if not file: return  words

    with open(file, "r") as f:
        raw = f.read()

        # 分割句子,并小写化
        sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sentence_tokenizer.tokenize(raw.lower())

        # NLTK分词:使用正则表达式
        # words = [nltk.word_tokenize(sentence) for sentence in sentences]
        # words =  [nltk.tokenize.WordPunctTokenizer().tokenize(sentence) for sentence in sentences]
        words = [nltk.regexp_tokenize(sentence, pattern) for sentence in sentences]
        # 列表扁平化
        words =  list(itertools.chain.from_iterable(words))

        # 过滤掉非字母
        filterers = string.ascii_lowercase + '.' + '-'
        words = [word for word in words if all(char in filterers for char in word) and word != '.']
        words = list(map(lambda word: word.replace('.', '').split("-"), words))
        words = list(itertools.chain.from_iterable(words))

    return words


def stemming(words):
    '''
    提取词干,还原
    :param words: 单词列表
    :return: 单词列表
    '''
    #stemmer = LancasterStemmer()
    # stemmer = SnowballStemmer("english")
    #stemmer = PorterStemmer()
    #lemmatizer = WordNetLemmatizer()
    #return [lemmatizer.lemmatize(stemmer.stem(word)) for word in words]
    #return [stemmer.stem(word) for word in words]

    # WorldNet同义词转换，初步提取


    ret = []
    lemmatizer = WordNetLemmatizer()
    for word in words:
        wordn = wn.morphy(word)
        wordn = wordn if wordn else word
        ret.append(lemmatizer.lemmatize(wordn))



    # 使用 SnowballStemmer 词干提取算法
    #stemmer = SnowballStemmer("english")
    #stemmer = LancasterStemmer()
    #stemmer = PorterStemmer()
    #ret = [stemmer.stem(word) for word in ret]

    return ret


def load_stopwords(file = " "):
    '''
    载入停用词，可指定停用词文件，否则使用默认文件
    :param file: 停用词文件
    :return: 停用词列表
    '''

    if file == " ":
        file = get_stopwords_file()

    data = []

    with open(file, 'rt') as f:
        for line in f:
            if len(line) != 0:
                data.append(line.split()[0])

    return data

def filter_stopwords(words):
    stopwords = load_stopwords()
    return [word for word in words if word not in stopwords and len(word) >= 2]

if __name__ == '_main__':
    tokenizing = token("data/ICML/1. Active Learning/Diagnosis determination.txt")
    #print(tokenizing)
    print(stemming(tokenizing))