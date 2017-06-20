#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/19 17:18
@annotation = ''
"""
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

data = pd.read_csv('basketball_shot_log.csv')
predictors = data.drop(['shot_result'], axis=1).as_matrix()
target = to_categorical(data.shot_result)
n_cols = predictors.shape[1]
model = Sequential()

model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation="softmax"))

"""
Stochastic gradient descent
"""
# my_optimizer = SGD(lr=lr)

"""
Add metrics = [‘accuracy’] to compile step for easy-tounderstand
diagnostics
"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(predictors, target)
from keras.models import load_model

model.save('model_file.h5')
model = load_model('my_model.h5')
predictions = model.predict(data_to_predict_with)
probability_true = predictions[:, 1]

print model.summary()

"""
Validation in deep learning
● Commonly use validation split rather than crossvalidation
● Deep learning widely used on large datasets
● Single validation score is based on large amount of
data, and is reliable
● Repeated training from cross-validation would take
long time
"""
# model.fit(predictors, target, validation_split=0.3)

"""
Early Stopping
"""
from keras.callbacks import EarlyStopping

early_stopping_monitor = EarlyStopping(patience=2)

model.fit(predictors, target, validation_split=0.3, epochs=20, callbacks=[early_stopping_monitor])

"""
optimizing model capacity

● Start with a small network
● Gradually increase capacity
● Keep increasing capacity until validation score is no longer
"""


