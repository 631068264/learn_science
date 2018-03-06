#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/30 17:08
@annotation = ''
"""
import mglearn
from sklearn.linear_model import Ridge, LinearRegression, Lasso
from sklearn.model_selection import train_test_split

"""线性回归"""

X, y = mglearn.datasets.load_extended_boston()
# X, y = mglearn.datasets.make_wave(n_samples=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
print X_train.shape
print X_test.shape

"""
减少系数对outcome影响 贴近0 解决过拟合

过拟合 : 早停法 加入对模型复杂度惩罚

正则化 Regularization
    Regularization means explicitly restricting a model to avoid overfitting.
L2 = W1^2 + W2^2+ ...+Wn^2
The particular kind used by ridge regression is known as L2 regularization
执行 L2 正则化对模型具有以下影响
    使权重值接近于 0（但并非正好为 0）
    使权重的平均值接近于 0，且呈正态（钟形曲线或高斯曲线）分布




Increasing alpha forces coefficients to move more toward zero, 
which decreases training set performance but might help generalization.
Decreasing alpha allows the coefficients to be less restricted

Lasso 与 Ridge 相似使用L1 regularization (some coefficients are exactly zero)

To reduce underfitting, let’s try decreasing alpha
When we do this, we also need to increase the default 
setting of max_iter (the maximum number of iterations to run)

The Ridge model with alpha=0.1 has similar predictive performance as the lasso model with alpha=0.01, 
but using Ridge, all coef‐ ficients are nonzero


scikit-learn also provides the ElasticNet class, which combines the penalties of Lasso and Ridge. In practice, 
this combination works best, 
though at the price of having two parameters to adjust: one for the L1 regularization, and one for the L2 regularization.
"""
lr = Ridge().fit(X_train, y_train)
lr = LinearRegression().fit(X_train, y_train)
lr = Lasso().fit(X_train, y_train)

print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

#
# lr = LinearRegression().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#
# ridge = Ridge().fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
#
# ridge10 = Ridge(alpha=10).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))
#
# ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
# print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
# print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))
# plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
# plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
# plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
# plt.plot(lr.coef_, 'o', label="LinearRegression")
# plt.xlabel("Coefficient index")
# plt.ylabel("Coefficient magnitude")
# plt.hlines(0, 0, len(lr.coef_))
# plt.ylim(-25, 25)
# plt.legend()
# plt.show()

"""
plots that show model performance 
as a function of dataset size are called learning curves
学习曲线
"""
# mglearn.plots.plot_ridge_n_samples()
# plt.show()

"""
多元回归

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



X = [[50],[100],[150],[200],[250],[300]]
y = [[150],[200],[250],[280],[310],[330]]
X_test = [[250],[300]] # 用来做最终效果测试
y_test = [[310],[330]] # 用来做最终效果测试
plt.plot(X, y, 'k.')

model = LinearRegression()
model.fit(X, y)
X2 = [[30], [400]]
y2 = model.predict(X2)
plt.plot(X2, y2, 'g-')

xx = np.linspace(30, 400, 100) # 设计x轴一系列点作为画图的x点集
quadratic_featurizer = PolynomialFeatures(degree=2) # 实例化一个二次多项式特征实例
X_train_quadratic = quadratic_featurizer.fit_transform(X) # 用二次多项式对样本X值做变换
xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1)) # 把训练好X值的多项式特征实例应用到一系列点上,形成矩阵
regressor_quadratic = LinearRegression() # 创建一个线性回归实例
regressor_quadratic.fit(X_train_quadratic, y) # 以多项式变换后的x值为输入，代入线性回归模型做训练
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-') # 用训练好的模型作图

print '一元线性回归 r-squared', model.score(X_test, y_test)
X_test_quadratic = quadratic_featurizer.transform(X_test)
print '二次回归     r-squared', regressor_quadratic.score(X_test_quadratic, y_test)

plt.show() # 展示图像

"""
