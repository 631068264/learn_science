#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/19 09:31
@annotation = ''
"""
import pandas as pd

X_train, X_test, y_train, y_test = multilabel_train_test_split(X, y, size=0.2, seed=123)

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

"""
多元逻辑回归 one vs other
"""
clf = OneVsRestClassifier(LogisticRegression())
clf.fit(X_train, y_train)

"""
If .predict() was used instead:
● Output would be 0 or 1
● Log loss penalizes being confident and wrong
● Worse performance compared
to .predict_proba()
"""
predictions = clf.predict_proba(holdout)
prediction_df = pd.DataFrame(columns=pd.get_dummies(df[LABELS],
                            prefix_sep='__').columns,
                            index=holdout.index,
                            data=predictions)

prediction_df.to_csv('predictions.csv')
score = score_submission(pred_path='predictions.csv')