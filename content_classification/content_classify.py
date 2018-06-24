#coding: utf-8
#ref: https://blog.csdn.net/u012871493/article/details/72782744
import sys
import os
import pandas as pd
import numpy as np
import cPickle
import time

import jieba # pip install jieba

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dropout, Dense, Activation

from sklearn.metrics import precision_score
from sklearn.metrics import classification_report


reload(sys)
sys.setdefaultencoding('utf-8')

timeStamp_model = str(int(time.time())) + "_model" #保存模型
os.mkdir(timeStamp_model)


neg=pd.read_excel('dataset/neg.xls',header=None,index=None) #len: 10428
pos=pd.read_excel('dataset/pos.xls',header=None,index=None) #len: 10677

#add label
neg['label'] = 0
pos['label'] = 1

pos_neg = pd.concat([pos, neg], ignore_index=True) # ignore_index 合并时， 忽略index， 顺序递增即可 #len: 21105 #reset_index也能解决index合并递增

cut_func = lambda x: list(jieba.cut(x))
pos_neg['words'] = pos_neg[0].apply(cut_func)
# apply的第一个参数，函数，相当于C/C++的函数指针
# axis = 1，就会把一行； 默认为axis = 0，列，数据作为Series的数据 结构传入给自己实现的函数中； 自动遍历 很高效

extra_corpus = pd.read_excel('dataset/sum.xls')  # 保留 原始header # len: 12895
extra_corpus = extra_corpus[extra_corpus['rateDate'].notnull()] 
# ‘sum.xls’数据‘串列’，‘rateDate’列中实际存放的是评论信息，理应为‘rateContent’ # len: 12838
# extra_corpus['rateDate'].notnull() 返回一个 bool Series(一组数据: 索引在左边，值在右边; 可以看成字典)， 是否为空
extra_corpus['words'] = extra_corpus['rateDate'].apply(cut_func)

#print extra_corpus
pos_neg_extra_words = pd.concat([pos_neg['words'], extra_corpus['words']], ignore_index=True) #len: 33942

all_words = []
for i in pos_neg_extra_words:
	all_words.extend(i)
#len(all_words) #1509205 #包含重复的所有词
all_words = pd.DataFrame(pd.Series(all_words).value_counts()) #len: 51301, 自动倒序排列 # all_words的最左侧索引就是词本身
all_words['id'] = range(len(all_words))

get_sent_func = lambda x: list(all_words['id'][x])
pos_neg['sent'] = pos_neg['words'].apply(get_sent_func)  #需要一段时间
#把[美的, 售后, 太, 垃圾, ，, 其他, 售后, 都, 是, 两, 小时, 回, 电话,... 映射成 [745, 623, 88, 714, 0, 238, 623, 15, 7, 1894, ...

print "processing: pad sequences ---------- ----------"
maxlen = 50
pos_neg['sent'] = list(sequence.pad_sequences(pos_neg['sent'], maxlen=maxlen)) 
#长于该长度maxlen 的序列将会截短，短于该长度的序列将会填充 #keras只能接受长度相同的序列输入
#pad_sequences： 默认参数： padding='pre', truncating='pre'，返回 二维numpy  
#通过list(把二维ndarray编程[一维array1， 一维array2]) 交给 pos_neg['sent']（DF的列）


pos_neg = pos_neg.sample(frac=1.0, random_state=0).reset_index(drop=True) #丢弃原来的（sample乱序之前的）序号， 维持序号递增
#.reset_index(drop=True)  


trainAnaTestX = np.array(list(pos_neg['sent']))
trainAnaTestY = np.array(list(pos_neg['label']))
trainX = np.array(list(pos_neg['sent']))
trainY = np.array(list(pos_neg['label']))
testX = np.array(list(pos_neg['sent']))
testY = np.array(list(pos_neg['label']))
cPickle.dump(testX, open(timeStamp_model + "/testX.pkl", "wb"))
cPickle.dump(testY, open(timeStamp_model + "/testY.pkl", "wb"))
print trainAnaTestX
print trainAnaTestY
print trainX

#快速开始序贯（Sequential）模型:http://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/
print "processing: build model ---------- ----------"
model = Sequential()
#嵌入层 Embedding: http://keras-cn.readthedocs.io/en/latest/layers/embedding_layer/   
#Embedding 理解， 类似 word embedding词向量: http://blog.sina.com.cn/s/blog_1450ac3c60102x79x.html
model.add(Embedding(len(all_words) + 1, 256))
model.add(LSTM(128)) #输入256， 输出128-dim
#http://keras-cn.readthedocs.io/en/latest/layers/core_layer/
model.add(Dropout(0.5, seed=0))
#Dense就是常用的全连接层
model.add(Dense(1)) #输入128， 输出1-dim
model.add(Activation('sigmoid'))

#模型编译： http://keras-cn.readthedocs.io/en/latest/getting_started/sequential_model/#_1
#class_mode： 该参数决定了返回的标签数组的形式, “categorical”会返回2D的one-hot编码标签,”binary”返回1D的二值标签.”sparse”返回1D的整数标签 
#2.0.9版本keras不支持class_mode：
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

#http://keras-cn.readthedocs.io/en/latest/models/sequential/     fit
model.fit(trainX, trainY, batch_size=128, epochs=10)  #nb_epoch or epochs， 和版本有关

predY = model.predict_classes(testX)
print "precisionScore : " + str(precision_score(testY, predY))
#classification_report: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
print str(classification_report(testY, predY))



model.save(timeStamp_model + '/lstmModel.h5')
#model = load_model('lstmModel.h5') 
