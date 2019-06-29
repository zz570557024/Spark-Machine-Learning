from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn
import gensim
import re
from collections import namedtuple
from gensim.models import Doc2Vec
from gensim.models.doc2vec import Doc2Vec,LabeledSentence
from random import shuffle
import multiprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

max_features=5000
max_document_length=1000
vocabulary=None
doc2ver_bin="doc2ver.bin"
word2ver_bin="word2ver.bin"
#LabeledSentence = gensim.models.doc2vec.LabeledSentence
SentimentDocument = namedtuple('SentimentDocument', 'words tags')




def load_one_file(filename):
    x=""
    with open(filename,errors="ignore") as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    f.close()
    return x

def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    x_train=[]
    y_train=[]
    x_test=[]
    y_test=[]
    path="./data/review/aclImdb/train/pos/"
    print ("Load %s" % path)
    x_train=load_files_from_dir(path)
    y_train=[0]*len(x_train)
    path="./data/review/aclImdb/train/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_train+=[1]*len(tmp)
    x_train+=tmp

    path="./data/review/aclImdb/test/pos/"
    print ("Load %s" % path)
    x_test=load_files_from_dir(path)
    y_test=[0]*len(x_test)
    path="./data/review/aclImdb/test/neg/"
    print ("Load %s" % path)
    tmp=load_files_from_dir(path)
    y_test+=[1]*len(tmp)
    x_test+=tmp

    return x_train, x_test, y_train, y_test

# 使用词袋模型提取特征
def get_features_by_wordbag():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    # data = open(r"G:\paper\2book-master\2book-master\code\out.txt", "w")  # txt格式输出分词结果
    vocabulary=vectorizer.vocabulary_
    # data.write(str(vocabulary))
    # data.close()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1 )
    print (vectorizer)
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test

def show_diffrent_max_features():
    global max_features
    a=[]
    b=[]
    for i in range(1000,20000,2000):
        max_features=i
        print ("max_features=%d" % i)
        x, y = get_features_by_wordbag()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
        gnb = GaussianNB()
        gnb.fit(x_train, y_train)
        y_pred = gnb.predict(x_test)
        score=metrics.accuracy_score(y_test, y_pred)
        a.append(max_features)
        b.append(score)
        plt.plot(a, b, 'r')
    plt.xlabel("max_features")
    plt.ylabel("metrics.accuracy_score")
    plt.title("metrics.accuracy_score VS max_features")
    plt.legend()
    plt.show()

# 得出分类准确度表征量
def do_metrics(y_test,y_pred):
    print (metrics.accuracy_score(y_test,y_pred)) #分类准确率
    print (metrics.confusion_matrix(y_test,y_pred)) #混淆矩阵
    print(metrics.classification_report(y_test,y_pred)) #分类报告
    print(metrics.hamming_loss(y_test,y_pred)) #平均汉明距离或平均Hamming loss
    print(metrics.precision_score(y_test,y_pred)) #精确率
    print(metrics.recall_score(y_test,y_pred)) #召回率
    print(metrics.f1_score(y_test,y_pred)) #F值
    print(metrics.precision_recall_curve(y_test,y_pred)) #不同概率阀值的precision-recall对
    # 画ROC曲线
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('Receiver operating characteristic example')
    plt.show()

# 朴素贝叶斯算法+词袋模型
def do_nb_wordbag(x_train, x_test, y_train, y_test):
    print ("NB and wordbag")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    do_metrics(y_test,y_pred)

# 朴素贝叶斯算法+doc2vec模型
def do_nb_doc2vec(x_train, x_test, y_train, y_test):
    print ("NB and doc2vec")
    gnb = GaussianNB()
    gnb.fit(x_train,y_train)
    y_pred=gnb.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

# 支持向量机算法+词袋模型
def do_svm_wordbag(x_train, x_test, y_train, y_test):
    print ("SVM and wordbag")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

# 支持向量机算法+doc2vec模型
def do_svm_doc2vec(x_train, x_test, y_train, y_test):
    print ("SVM and doc2vec")
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

# 决策树模型+doc2vec模型
def do_rf_doc2vec(x_train, x_test, y_train, y_test):
    print ("rf and wordbag")
    clf = RandomForestClassifier(n_estimators=10) #随机树个数
    clf.fit(x_train, y_train) #训练集训练
    y_pred = clf.predict(x_test)
    do_metrics(y_test,y_pred)

# 使用词袋模型+tfidf提取特征
def get_features_by_wordbag_tfidf():
    global max_features
    x_train, x_test, y_train, y_test=load_all_files()

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 max_features=max_features,
                                 stop_words='english',
                                 max_df=1.0,
                                 min_df=1,
                                 binary=True)
    print (vectorizer)
    x_train=vectorizer.fit_transform(x_train)
    x_train=x_train.toarray()
    vocabulary=vectorizer.vocabulary_

    vectorizer = CountVectorizer(
                                 decode_error='ignore',
                                 strip_accents='ascii',
                                 vocabulary=vocabulary,
                                 stop_words='english',
                                 max_df=1.0,binary=True,
                                 min_df=1 )
    print (vectorizer)
    x_test=vectorizer.fit_transform(x_test)
    x_test=x_test.toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x_train=transformer.fit_transform(x_train)
    x_train=x_train.toarray()
    x_test=transformer.transform(x_test)
    x_test=x_test.toarray()

    return x_train, x_test, y_train, y_test

# cnn+词袋模型
def do_cnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print ("CNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_document_length], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")

# cnn+doc2vec
def do_cnn_doc2vec_2d(trainX, testX, trainY, testY):
    print ("CNN and doc2vec 2d")
    #print ("CNN and wordbag")

    trainX = trainX.reshape([-1, max_features, max_document_length, 1])
    testX = testX.reshape([-1, max_features, max_document_length, 1])


    # Building convolutional network
    network = input_data(shape=[None, max_features, max_document_length, 1], name='input')
    network = conv_2d(network, 16, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy', name='target')

    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': trainX}, {'target': trainY}, n_epoch=20,
               validation_set=({'input': testX}, {'target': testY}),
               snapshot_step=100, show_metric=True, run_id='review')

# cnn+doc2vec
def do_cnn_doc2vec(trainX, testX, trainY, testY):
    global max_features
    print ("CNN and doc2vec")

    #trainX = pad_sequences(trainX, maxlen=max_features, value=0.)
    #testX = pad_sequences(testX, maxlen=max_features, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Building convolutional network
    network = input_data(shape=[None,max_features], name='input')
    network = tflearn.embedding(network, input_dim=1000000, output_dim=128,validate_indices=False)
    branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = global_max_pool(network)
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY,
              n_epoch=5, shuffle=True, validation_set=(testX, testY),
              show_metric=True, batch_size=100,run_id="review")

# rnn+词袋模型
def do_rnn_wordbag(trainX, testX, trainY, testY):
    global max_document_length
    print ("RNN and wordbag")

    trainX = pad_sequences(trainX, maxlen=max_document_length, value=0.)
    testX = pad_sequences(testX, maxlen=max_document_length, value=0.)
    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=2)
    testY = to_categorical(testY, nb_classes=2)

    # Network building
    net = tflearn.input_data([None, max_document_length])
    net = tflearn.embedding(net, input_dim=640000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=10,run_id="review",n_epoch=5)
    
    predY = model.predict(testX)
    do_metrics(testX,predY)

# dnn+词袋模型
def do_dnn_wordbag(x_train, x_test, y_train, y_test):
    print ("MLP and wordbag")
    # print ("MLP and wordbag&TF-IDF")
    # Building deep neural network
    clf = MLPClassifier(solver='adam',
                        alpha=1e-5,
                        hidden_layer_sizes = (5, 2),
                        random_state = 1)
    print  (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    # print (metrics.accuracy_score(y_test, y_pred))
    # print (metrics.confusion_matrix(y_test, y_pred))
    do_metrics(y_test,y_pred)

# dnn+doc2vec
def do_dnn_doc2vec(x_train, x_test, y_train, y_test):
    print ("MLP and doc2vec")
    global max_features
    # Building deep neural network
    clf = MLPClassifier(solver='adam', # 默认 ‘adam’，用来优化权重 
                        alpha=1e-5, # float,可选的，默认0.0001,正则化项参数 
                        hidden_layer_sizes = (5, 2), #元祖格式，长度=n_layers-2, 默认(100，），第i个元素表示第i个隐藏层的神经元的个数
                        random_state = 1) # int 或RandomState，可选，默认None，随机数生成器的状态或种子。
    print  (clf)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print (metrics.accuracy_score(y_test, y_pred))
    print (metrics.confusion_matrix(y_test, y_pred))

# 使用tf提取特征
def  get_features_by_tf():
    global  max_document_length
    x_train, x_test, y_train, y_test=load_all_files()

    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x_train=vp.fit_transform(x_train, unused_y=None)
    x_train=np.array(list(x_train))

    x_test=vp.transform(x_test)
    x_test=np.array(list(x_test))
    return x_train, x_test, y_train, y_test

# 清洗数据
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n', '') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    # treat punctuation as individual words
    for c in punctuation:
        corpus = [z.replace(c, ' %s ' % c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus

# 正常化文本
def normalize_text(text):
    norm_text = text.lower()

    # Replace breaks with spaces
    norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text

def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        #labelized.append(LabeledSentence(v, [label]))
        #labelized.append(LabeledSentence(words=v,tags=label))
        labelized.append(SentimentDocument(v, [label]))
    return labelized

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.array(np.concatenate(vecs),dtype='float')


def getVecsByWord2Vec(model, corpus, size):
    global max_document_length
    #x=np.zeros((max_document_length,size),dtype=float, order='C')
    x=[]

    for text in corpus:
        xx = []
        for i, vv in enumerate(text):
            try:
                xx.append(model[vv].reshape((1,size)))
            except KeyError:
                continue

        x = np.concatenate(xx)

    x=np.array(x, dtype='float')
    return x

# 使用doc2vec提取向量
def  get_features_by_doc2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x_train = labelizeReviews(x_train, 'TRAIN')
    x_test = labelizeReviews(x_test, 'TEST')

    x=x_train+x_test
    cores=multiprocessing.cpu_count()
    #models = [
        # PV-DBOW
    #    Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores),
        # PV-DM w/average
    #    Doc2Vec(dm=1, dm_mean=1, size=200, window=8, min_count=19, iter=10, workers=cores),
    #]
    if os.path.exists(doc2ver_bin):
        print ("Find cache file %s" % doc2ver_bin)
        model=Doc2Vec.load(doc2ver_bin)
    else:
        model=Doc2Vec(dm=0, size=max_features, negative=5, hs=0, min_count=2, workers=cores,iter=60)


        #for model in models:
        #    model.build_vocab(x)
        model.build_vocab(x)

        #models[1].reset_from(models[0])

        #for model in models:
        #    model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        #models[0].train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(doc2ver_bin)

    #x_test=getVecs(models[0],x_test,max_features)
    #x_train=getVecs(models[0],x_train,max_features)
    x_test=getVecs(model,x_test,max_features)
    x_train=getVecs(model,x_train,max_features)

    return x_train, x_test, y_train, y_test

# 使用word2vec提取向量
def  get_features_by_word2vec():
    global  max_features
    x_train, x_test, y_train, y_test=load_all_files()

    x_train=cleanText(x_train)
    x_test=cleanText(x_test)

    x=x_train+x_test
    cores=multiprocessing.cpu_count()

    if os.path.exists(word2ver_bin):
        print ("Find cache file %s" % word2ver_bin)
        model=gensim.models.Word2Vec.load(word2ver_bin)
    else:
        model=gensim.models.Word2Vec(size=max_features, window=5, min_count=10, iter=10, workers=cores)

        model.build_vocab(x)

        model.train(x, total_examples=model.corpus_count, epochs=model.iter)
        model.save(word2ver_bin)


    x_train=getVecsByWord2Vec(model,x_train,max_features)
    x_test = getVecsByWord2Vec(model, x_test, max_features)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    # print ("Hello review")
    ################################################
    # demo1
    print ("get_features_by_wordbag")
    # x_train, x_test, y_train, y_test=get_features_by_wordbag()
    # print ("get_features_by_word2vec")
    # print ("get_features_by_doc2vec")
    x_train, x_test, y_train, y_test=get_features_by_doc2vec()
    # x_train, x_test, y_train, y_test=get_features_by_word2vec()
    # x_train, x_test, y_train, y_test=get_features_by_wordbag_tfidf()
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    do_dnn_wordbag(x_train, x_test, y_train, y_test)
    # do_rf_doc2vec(x_train, x_test, y_train, y_test)
    # do_cnn_wordbag(x_train, x_test, y_train, y_test)
    # do_rnn_wordbag(x_train, x_test, y_train, y_test)
    # do_nb_wordbag(x_train, x_test, y_train, y_test)
    # demo2
    # print ("get_features_by_wordbag")
    # x_train, x_test, y_train, y_test=get_features_by_wordbag()
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    # demo3
    # print ("get_features_by_wordbag_tfidf")
    # x_train, x_test, y_train, y_test=get_features_by_wordbag_tfidf()
    # do_nb_wordbag(x_train, x_test, y_train, y_test)
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    # demo4
    # print ("get_features_by_word2vec")
    # x_train, x_test, y_train, y_test=get_features_by_word2vec()
    # do_nb_wordbag(x_train, x_test, y_train, y_test)
    # do_svm_wordbag(x_train, x_test, y_train, y_test)
    # demo5
   
