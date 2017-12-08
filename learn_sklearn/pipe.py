#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/5 17:42
@annotation = ''
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, decomposition, datasets
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

logistic = linear_model.LogisticRegression()

pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
if True:
    """
    We have seen that some estimators can transform data and that some estimators can predict variables.
    We can also create combined estimators
    """
    digits = datasets.load_digits()
    X_digits = digits.data
    y_digits = digits.target

    ###############################################################################
    # Plot the PCA spectrum
    pca.fit(X_digits)

    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_')

    ###############################################################################
    # Prediction

    n_components = [20, 40, 64]
    Cs = np.logspace(-4, 4, 3)

    # Parameters of pipelines can be set using ‘__’ separated parameter names:

    estimator = GridSearchCV(pipe,
                             dict(pca__n_components=n_components,
                                  logistic__C=Cs))
    estimator.fit(X_digits, y_digits)

    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    plt.legend(prop=dict(size=12))
    plt.show()

if False:
    from sklearn.preprocessing import FunctionTransformer

    from sklearn.pipeline import FeatureUnion

    """
    Turns a Python function into an object that a scikit-learn pipeline can understand
    """
    get_text_data = FunctionTransformer(lambda x: x['text'],
                                        validate=False)
    get_numeric_data = FunctionTransformer(lambda x: x[['numeric',
                                                        'with_missing']], validate=False)
    from sklearn.pipeline import FeatureUnion

    # numeric_pipeline = Pipeline([
    #     ('selector', get_numeric_data),
    #     ('imputer', Imputer())
    # ])
    # text_pipeline = Pipeline([
    #     ('selector', get_text_data),
    #     ('vectorizer', CountVectorizer())
    # ])
    # pl = Pipeline([
    #     ('union', FeatureUnion([
    #         ('numeric', numeric_pipeline),
    #         ('text', text_pipeline)
    #     ])),
    #     ('clf', OneVsRestClassifier(LogisticRegression()))
    # ])

    pl = Pipeline([
        ('union', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('imputer', Imputer())
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vectorizer', CountVectorizer())
            ]))
        ])
         ),
        ('clf', OneVsRestClassifier(LogisticRegression()))
    ])
