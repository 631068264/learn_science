#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/7/29 21:43
@annotation = ''
"""
"""
StandardScaler 
    (y-mean)/scale
    in scikit-learn ensures that for each feature the mean is 0 and the variance is 1, 
    bringing all features to the same magnitude
    
    this scaling does not ensure any particular minimum and maximum values for the features
    
MinMaxScaler
    this means all of the data is contained within the rectangle created by the x-axis between 0 and 1 
    and the y-axis between 0 and 1
    
Normalizer
    In other words, it projects a data point on the circle (or sphere, in the case of higher dimensions) 
    with a radius of 1
    
MaxAbsScaler    
适合稀疏数据缩放

数据归一化/规范化（Normalization）
规范化是使单个样本具有单位范数的缩放操作

数值特征二值化（Binarization）
数值型特征变成布尔型特征

类别数据编码 OneHot 编码
将具有多个类别的特征转换为多维二元特征，所有二元特征互斥，当某个二元特征为 1 时，表示取某个类别

from sklearn import preprocessing
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()  #array([[ 1., 0., 0., 1., 0., 0., 0., 0., 1.]])

from sklearn.preprocessing import OneHotEncoder
onehot=[]
for i in xrange(train_all.shape[0]) :
  tmp=[]
  tmp.append(train_all["cate_1_name"][i])
  tmp.append(train_all["cate_2_name"][i])
  tmp.append(train_all["cate_3_name"][i])
  onehot.append(tmp)
enc = OneHotEncoder()
enc.fit(onehot)
onehot_feature=enc.transform(onehot).toarray()
onehot_feature=pd.DataFrame({'onehot':onehot})


标签二值化
类别特征转换为多维二元特征，并将每个特征扩展成用一维表示 
from sklearn.preprocessing import label_binarize
label_binarize([1, 6], classes=[1, 2, 4, 6])

类别编码
from sklearn import preprocessing
tmp=['A','A','b','c']
le = preprocessing.LabelEncoder()
le.fit(tmp)
le.transform(tmp)


缺失值处理
使用缺失数值所在行或列的均值、中位数、众数来替代缺失值 
preprocessing.Imputer(missing_values=’NaN’, strategy=’mean’, axis=0, verbose=0, copy=True)

生成多项式特征
preprocessing.PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)


模型保存

使用 joblib 保存,读取速度也相对pickle快
from sklearn.externals import joblib #jbolib模块
#保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')
#读取Model
clf3 = joblib.load('save/clf.pkl') 

"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,
                                                    random_state=1)
# print(X_train.shape)
# print(X_test.shape)

if False:
    scaler = MinMaxScaler()

    print X_train
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # same result, but more efficient computation
    # X_scaled_d = scaler.fit_transform(X)

    print X_train_scaled
if False:
    import numpy as np

    y = [1, 2, 4, 6]
    y_label_binarize = label_binarize(y, classes=y)
    print y_label_binarize
    label_cnt = len(np.unique(y))
    for ind in np.arange(0, label_cnt):
        # 开始 one vs rest
        _y = y_label_binarize[:, ind]
        print _y