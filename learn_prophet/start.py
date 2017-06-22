#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/22 09:42
@annotation = ''
"""
import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plot

df = pd.read_csv('example_wp_peyton_manning.csv')
df.rename(columns={'ds': 'date'}, inplace=True)
df['y'] = np.log(df['y'])
print df.head()
print df.tail()
"""
changepoint_prior_scale Increasing it will make the trend more flexibile 趋势变化幅度大

m = Prophet(changepoints=['2014-01-01'])
forecast = m.fit(df).predict(future)
m.plot(forecast);

"""
m = Prophet(changepoint_prior_scale=0.1)
m.fit(df)

future = m.make_future_dataframe(periods=365)
# print future.tail()

forecast = m.predict(future)

print forecast.head()

# m.plot(forecast)
m.plot_components(forecast)

plot.show()
