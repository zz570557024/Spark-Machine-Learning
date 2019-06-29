# Spark机器学习

## Apache Spark

Apache Spark是一个开源集群运算框架，最初是由加州大学伯克利分校AMPLab所研发。

Spark核心模块：

* Spark Core
* Spark SQL
* Spark Streaming
* GraphX
* Mllib
* SparkR

## 机器学习

### 特征提取模型

* 词袋模型（Bag of words）
* 词频与逆向文件频率模型(TF-IDF)
* 句向量模型(Doc2Vec)

### 机器学习算法

* 朴素贝叶斯算法
* 决策树算法
* 支持向量机
* 多层感知机算法

### 机器学习工具

* Tensorflow
* Scikit-Learn

## 基于Doc2Vec的SVM算法

Doc2Vec在Word2Vec的基础上增加了一个段落向量。Doc2Vec能够训练出每一篇文本的文本向量，相比于Word2Vec模型，Doc2Vec模型能够更全面地理解文本地语义特征。

同时，由于SVM具有良好的泛化性能，且通过核函数可以处理低维线性不可分的情况，避开高维空间的复杂性，直接用内积函数，很适合对评论文本进行分类，所以本文将基于Doc2Vec的SVM算法引入到恶意评论检测中。

## 实验数据集

实验数据集来自于斯坦福大学AI实验室开源的互联网电影资料库（Internet Movie Database, IMDb）评论数据集。

## 实验评价标准

* 准确率（训练样本中被正确分类的数目除以总样本数）`Acc = (TP+TN)/(TP+TN+FP+FN)`
* 精确性（预测为正的样本中有多少是正确的）`Prc = TP/(TP+FP)`
* 召回率(样本中的正例有多少被预测正确了)`Sn = TP/(TP+FN)`
* F值（精确性和召回率的组合）`FScore = (2*Sn*Prc)/(Sn*Prc)`

## Spark评论检测系统架构

* Ubuntu 16.04.5 LTS
* Hadoop-2.7.7
* Spark-2.2.0-bin-hadoop2.7
* jdk_1.8.011-1
* Scala-2.11.8
* IDEA

