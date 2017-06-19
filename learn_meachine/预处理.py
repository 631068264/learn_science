#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/18 14:27
@annotation = ''
"""
"""
文字信息 转化为数字
"""
"""
Dealing with categorical features in Python
● scikit-learn: OneHotEncoder()
● pandas: get_dummies()
"""

"""
In [3]: df_origin = pd.get_dummies(df)
In [4]: print(df_origin.head())
 mpg displ hp weight accel size origin_Asia origin_Europe \
0 18.0 250.0 88 3139 14.5 15.0 0 0
1 9.0 304.0 193 4732 18.5 20.0 0 0
2 36.1 91.0 60 1800 16.4 10.0 1 0
3 18.5 250.0 98 3525 19.0 15.0 0 0
4 34.3 97.0 78 2188 15.8 10.0 0 1

"""

"""
Using the mean of the non-missing entries
In [1]: from sklearn.preprocessing import Imputer
In [2]: imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
In [3]: imp.fit(X)
In [4]: X = imp.transform(X)
"""

"""
Why scale your data?
● Many models use some form of distance to inform them
● Features on larger scales can unduly influence the model
● Example: k-NN uses distance explicitly when making predictions
● We want features to be on a similar scale
● Normalizing (or scaling and centering)

Ways to normalize your data
● Standardization: Subtract the mean and divide by variance
● All features are centered around zero and have variance one
● Can also subtract the minimum and divide by the range
● Minimum zero and maximum one
● Can also normalize so the data ranges from -1 to +1
● See scikit-learn docs for further details
"""

"""
In [2]: from sklearn.preprocessing import scale
In [3]: X_scaled = scale(X)
In [4]: np.mean(X), np.std(X)
Out[4]: (8.13421922452, 16.7265339794)
In [5]: np.mean(X_scaled), np.std(X_scaled)
Out[5]: (2.54662653149e-15, 1.0)
"""
