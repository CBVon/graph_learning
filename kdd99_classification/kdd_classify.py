#coding: utf-8
import sys
import numpy as np
import pandas as pd
import cPickle
import time

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
import xgboost
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier

from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

import cost_based_scoring

label_map_dict = {
    'normal.': 0,
    'ipsweep.': 1,
    'mscan.': 1,
    'nmap.': 1,
    'portsweep.': 1,
    'saint.': 1,
    'satan.': 1,
    'apache2.': 2,
    'back.': 2,
    'mailbomb.': 2,
    'neptune.': 2,
    'pod.': 2,
    'land.': 2,
    'processtable.': 2,
    'smurf.': 2,
    'teardrop.': 2,
    'udpstorm.': 2,
    'buffer_overflow.': 3,
    'loadmodule.': 3,
    'perl.': 3,
    'ps.': 3,
    'rootkit.': 3,
    'sqlattack.': 3,
    'xterm.': 3,
    'ftp_write.': 4,
    'guess_passwd.': 4,
    'httptunnel.': 3,  # disputation resolved
    'imap.': 4,
    'multihop.': 4,  # disputation resolved
    'named.': 4,
    'phf.': 4,
    'sendmail.': 4,
    'snmpgetattack.': 4,
    'snmpguess.': 4,
    'worm.': 4,
    'xlock.': 4,
    'xsnoop.': 4,
    'spy.': 4,
    'warezclient.': 4,
    'warezmaster.': 4  # disputation resolved
    }


"""
def merge_sparse_feature(df): #观测规律；多余操作； 可以避免意外
    df.loc[(df['service'] == 'ntp_u') #条件过滤， 取索引
    | (df['service'] == 'urh_i')
    | (df['service'] == 'tftp_u')
    | (df['service'] == 'red_i')
    , 'service'] = 'normal_service_group'

    df.loc[(df['service'] == 'pm_dump') 
    | (df['service'] == 'http_2784')
    | (df['service'] == 'harvest')
    | (df['service'] == 'aol')
    | (df['service'] == 'http_8001')
    , 'service'] = 'satan_service_group'

    return df
"""


origin_feats = ["duration", "protocol_type", "service", "flag", "src_bytes", 
"dst_bytes", "land", "wrong_fragment", "urgent", "hot", 
"num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", 
"num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", 
"is_host_login", "is_guest_login", "count", "srv_count", "serror_rate", 
"srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
"srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", 
"dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",    "dst_host_rerror_rate", 
"dst_host_srv_rerror_rate", "label"]
train = pd.read_csv("dataset/kddcup.data_10_percent_corrected", sep=",", names=origin_feats)
print("train data loaded")
train.label = train.label.map(label_map_dict)
train_x = train.iloc[:, : -1]
train_y = train.iloc[:, -1: ]

test = pd.read_csv("dataset/corrected", sep=",", names=origin_feats)
print("test data loaded")
test.label = test.label.map(label_map_dict)
test_x = test.iloc[:, : -1]
test_y = test.iloc[:, -1: ] # 注意 有没最后一个‘：’区别很大


def one_hot(df):
    service_one_hot = pd.get_dummies(df["service"]) #实现one-hot
    df = df.drop('service', axis=1) #列删除
    df = df.join(service_one_hot)

    # 测试数据 的“service_one_hot”有“icmp”列（事实上icmp更应该属于“protocol_type_one_hot”）， 但是训练数据没“icmp列”； 防止意外， 故删之;
    # 为了顺利 进行“protocol_type_one_hot”的join操作，必须删除； 或者 采用 “service_one_hot”的“icmp”列 重命名的方式
    if 'icmp' in df.columns: # df.columns可以得到所有的列标签
        df = df.drop('icmp', axis=1)

    protocol_type_one_hot = pd.get_dummies(df["protocol_type"])
    df = df.drop('protocol_type', axis=1)
    df = df.join(protocol_type_one_hot)

    flag_type_one_hot = pd.get_dummies(df["flag"])
    df = df.drop('flag', axis=1)
    df = df.join(flag_type_one_hot)
    return df


#train_x = merge_sparse_feature(train_x) #20180523 验证，merge操作是多余的
#test_x = merge_sparse_feature(test_x)
train_x = one_hot(train_x)
test_x = one_hot(test_x)
#注意 one-hot 之后， train_x(118)和test_x(116)形状不一致了

train_x_feat_names = set(train_x.columns.values)
test_x_feat_names = set(test_x.columns.values)
add_to_train_x_feat_names = test_x_feat_names - train_x_feat_names
for i in add_to_train_x_feat_names:
    train_x[i] = 0.0

add_to_test_x_feat_names = train_x_feat_names - test_x_feat_names
for i in add_to_test_x_feat_names:
    test_x[i] = 0.0

# classify_feat selection， 通过cPickle进行封装
"""
selected_feat_names = set()
for i in xrange(10):
    tmp_set = set()

    rfc = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=-1, random_state=i * i)
    # n_jobs: If -1, then the number of jobs is set to the number of cores.
    # random_state: random number generator
    rfc.fit(train_x, train_y["label"].values)
    print "classify_feat selection " + str(i) + ", training finished"

    importances = rfc.feature_importances_
    indices = np.argsort(importances)[::-1] # descending order
    # argsort: 数组值从小到大的索引值 
    # [x:y:z]切片索引,x是左端,y是右端,z是步长,步长的负号就是反向,从右到左取值 
    for f in xrange(min(train_x.shape[1], 50)): # need roughly more than 40 features according to experiments # train_x.shape[1]: train_x列数
        tmp_set.add(train_x.columns[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, train_x.columns[indices[f]], importances[indices[f]]))
        #%-*s 代表输入一个字符串，-号代表左对齐、后补空白，*号代表对齐宽度由输入时确定
    if i == 0:
        selected_feat_names = tmp_set
    else:
        selected_feat_names &= tmp_set
print str(len(selected_feat_names)) + " features are selected"
cPickle.dump(selected_feat_names, open("selected_feat_names.pkl", "wb"))
"""

selected_feat_names = cPickle.load(open("selected_feat_names.pkl", "rb"))
train_x = train_x[list(selected_feat_names)] # 20180523验证： 特征选择 相比 不选择好太多
test_x = test_x[list(selected_feat_names)]


print "LogisticRegression : "
lr = LogisticRegression(random_state=0)
start_time = time.time()
lr = lr.fit(train_x, train_y["label"].values)
end_time = time.time()
print("LogisticRegression, training finished, using : %.2f s" % (end_time - start_time))
predict_y = lr.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


print "DecisionTreeClassifier : "
dtc = DecisionTreeClassifier(random_state=0)
start_time = time.time()
dtc = dtc.fit(train_x, train_y["label"].values)
end_time = time.time()
print("DecisionTreeClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = dtc.predict(test_x) # 预测结果单纯是个 list
# test_y 带表头的表结构
# test_y["label"] 不带表头的表结构， 带索引
# test_y["label"].values # .values将表结构转为 单纯list
#print classification_report(test_y["label"].values, predict_y)
#print metrics.f1_score(test_y["label"].values, predict_y) # 多分类的f1(全局 准确和召回)没太大价值
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False)) #show: 否是展示计算过程
print "---------- ----------"


print "RandomForestClassifier : "
rfc = RandomForestClassifier(random_state=0)
start_time = time.time()
rfc = rfc.fit(train_x, train_y["label"].values)
end_time = time.time()
print("RandomForestClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = rfc.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


"""
# GridSearchCV, 调参， 寻找最优化参数组合
print "RandomForestClassifier : (n_estimators=100, oob_score=True, random_state=0)" # 20180523 验证 给定初始化 稍稍好于 随机初始化
rfc = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0)
rfc = rfc.fit(train_x, train_y["label"].values)
print "RandomForestClassifier, training finished"
predict_y = rfc.predict(test_x)
print classification_report(test_y["label"].values, predict_y)
cost_based_scoring.score(test_y["label"].values, predict_y, show=True)
"""
"""
# optimize params， using GridSearch #20180523 验证gs结果是对训练集的最佳拟合， 对于测试集不一定好
print("RandomForestClassifier, grid search begin")
rfc = RandomForestClassifier(n_jobs=-1)
parameters = {
    'n_estimators': xrange(10, 200 + 10, 10),
    'criterion': ("gini", "entropy")
}
scorer = cost_based_scoring.scorer(show=True)
gscv = GridSearchCV(estimator=rfc, param_grid=parameters, scoring=scorer,cv=3, verbose=2, refit=False, n_jobs=1, return_train_score=False)
# cv: cross-validation generator,
# verbose: Controls the verbosity: the higher, the more messages.
# refit: Refit an estimator using the best found parameters on the whole dataset.
# return_train_score: If False, the cv_results_ attribute will not include training scores.
gscv.fit(train_x, train_y["label"].values)
print("optimization params:", gscv.best_params_['n_estimators'], gscv.best_params_['criterion'])
print("RandomForestClassifier, grid search finished")
'''
[Parallel(n_jobs=1)]: Done  60 out of  60 | elapsed:  2.5min finished
('optimization params:', 90, 'entropy')
'''
"""


# https://blog.csdn.net/zhaocj/article/details/51648966 # 和rfc很类似， 区别在于ET计算过程更极端：全局样本； 全局特征做随机
print "ExtraTreesClassifier : "
etc = ExtraTreesClassifier(random_state=0)
start_time = time.time()
etc = etc.fit(train_x, train_y["label"].values)
end_time = time.time()
print("ExtraTreesClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = etc.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


print "AdaBoostClassifier : "
ada = AdaBoostClassifier(random_state=0)
start_time = time.time()
ada = ada.fit(train_x, train_y["label"].values)
end_time = time.time()
print("AdaBoostClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = ada.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


print "GradientBoostingClassifier : "
gbdt = GradientBoostingClassifier(random_state=0)
start_time = time.time()
gbdt = gbdt.fit(train_x, train_y["label"].values)
end_time = time.time()
print("GradientBoostingClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = gbdt.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


print "XGBClassifier : "
xgb = XGBClassifier(random_state=0)
start_time = time.time()
xgb = xgb.fit(train_x, train_y["label"].values)
end_time = time.time()
print("XGBClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = xgb.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"



"""
# 自定义评估函数的 xgboost.train
def myFeval(y_pred, xgbTrainXy):
    y_pred = y_pred.astype(int)
    y_true = xgbTrainXy.get_label()
    y_true = y_true.astype(int) # numpy.float32 转化为 numpy.int

    cost_matrix = [[0, 1, 2, 2, 2],
                   [1, 0, 2, 2, 2],
                   [2, 1, 0, 2, 2],
                   [3, 2, 2, 0, 2],
                   [4, 2, 2, 2, 0]
                   ]
    cost = 0.0
    size = y_true.size

    for i in range(size):
        cost += cost_matrix[y_true[i]][y_pred[i]]
    return "myFeval", float(cost) / size


print "xgboost.train : "
#xgb = XGBClassifier(random_state=0) #初始化参数列表， 此处默认 + 随机初始化
xgbTrainXy = xgboost.DMatrix(train_x, train_y["label"].values) #xgbTrainXy.get_label() 结果是numpy.float32
xgbTestXy = xgboost.DMatrix(test_x, test_y["label"].values)
xgbTestX = xgboost.DMatrix(test_x)
start_time = time.time()
params = {
    'objective' : 'multi:softmax',  #多分类
    'num_class' : 5
}
xgb = xgboost.train(params=params, dtrain=xgbTrainXy, num_boost_round=10000, evals=[(xgbTrainXy, "trainSet"), (xgbTestXy, "testSet")], feval=myFeval, maximize=False, early_stopping_rounds=100)
# maximize=False ： 最小化评估函数
# 'testSet-myFeval' will be used for early stopping.  防止训练集过拟合
# early_stopping_rounds:要求evals 里至少有 一个元素，如果有多个，按最后一个去执行
end_time = time.time()
print("XGBClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = xgb.predict(xgbTestX) # predict_y predict_y
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y.astype(int), show=False)) 
print "---------- ----------"
'''
自定义评估函数 ， 20180524 验证 确实要比默认的 XGBClassifier要强
XGBClassifier, training finished, using : 106.78 s
score : 0.238820174325

最佳得分
Stopping. Best iteration:
[5] trainSet-merror:0.0005  testSet-merror:0.070244 trainSet-myFeval:0.000893   testSet-myFeval:0.230053
'''
"""


print "VotingClassifier : " 
#20180524 验证 几个差不多的模型集成投票时，投票模型稍稍好于被集成的模型； 一个好的 和 几个比较差的， 投票模型会稍稍差于 最好的那个（当然受调参影响）
vt = VotingClassifier(estimators=[('dtc', dtc), ('rfc', rfc), ('etc', etc)], voting='soft')
# weights 控制权重
start_time = time.time()
vt = vt.fit(train_x, train_y["label"].values)
end_time = time.time()
print("VotingClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = vt.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


np.random.seed(0) # 控制StackingCVClassifier随机因子， 类似于以上的 random_state=0


print "StackingCVClassifier : "
scvc = StackingCVClassifier(classifiers=[dtc, rfc, etc], meta_classifier=lr, use_probas=True, verbose=0)
# 关于数据的数据，一般是结构化数据（如存储在数据库里的数据，规定了字段的长度、类型等）
# meta_classifier ： 关于分类器的分类器，通常是主分类器的代理，用于提供附加的数据预处理
# use_probas : If True, trains meta-classifier based on predicted probabilities instead of class labels.
# verbose>2: Changes verbose param of the underlying regressor to self.verbose - 2  输出计算过程,赘言
start_time = time.time()
scvc = scvc.fit(train_x.values, train_y["label"].values) #stack对输入要求是numpy.array， 所以pandas.df必须转换，即.values
end_time = time.time()
print("StackingCVClassifier, training finished, using : %.2f s" % (end_time - start_time))
predict_y = scvc.predict(test_x)
#print classification_report(test_y["label"].values, predict_y)
print "score : " + str(cost_based_scoring.score(test_y["label"].values, predict_y, show=False))
print "---------- ----------"


'''
20180524 最终输出
train data loaded
test data loaded
LogisticRegression : 
LogisticRegression, training finished, using : 105.72 s
score : 0.491240366654
---------- ----------
DecisionTreeClassifier : 
DecisionTreeClassifier, training finished, using : 2.71 s
score : 0.23169543676
---------- ----------
RandomForestClassifier : 
RandomForestClassifier, training finished, using : 2.88 s
score : 0.246031720515
---------- ----------
ExtraTreesClassifier : 
ExtraTreesClassifier, training finished, using : 2.05 s
score : 0.246517205791
---------- ----------
AdaBoostClassifier : 
AdaBoostClassifier, training finished, using : 24.49 s
score : 0.388523256674
---------- ----------
GradientBoostingClassifier : 
GradientBoostingClassifier, training finished, using : 388.94 s
score : 0.226924177488
---------- ----------
XGBClassifier : 
XGBClassifier, training finished, using : 291.12 s
score : 0.240353793376
---------- ----------
VotingClassifier : 
VotingClassifier, training finished, using : 7.26 s
score : 0.236248066901
---------- ----------
StackingCVClassifier : 
StackingCVClassifier, training finished, using : 29.73 s
score : 0.240045140485
---------- ----------

'''
