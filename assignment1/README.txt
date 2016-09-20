1. 系统环境

   系统：Ubuntu16.04.1 64位 或 Windows 10
   IDE：PyCharm 2016.2
   编程语言：Python 3.5.2 （Anaconda Python 4.1.1）
   使用到的第三方Python库 ： NLTK3.2.1，需要下载语料库：python3 -c "import nltk;nltk.download_gui()"
   数据：原始数据在项目根目录下的data目录下，文件夹为ICML(即文件解压后的目录)，另外停用词放在data目录下的StopWords目录下

2. 运行

   1): 进入到code根目录,可看到项目文件
   2): 在终端中运行：python3 runAssignment1.py,等待4分钟左右后，待终端运行结束后便可以在assignment1/result目录下看到结果文件。结果文件共有16个文件，其中word_list.txt是“单词-序号”映射文件，其余的是每个类别产生的TF-IDF稀疏向量文件。
