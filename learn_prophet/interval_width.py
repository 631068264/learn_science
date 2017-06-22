#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/22 11:05
@annotation = ''
"""
import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plot

df = pd.read_csv('example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
print df.head()
print df.tail()

m = Prophet(changepoint_prior_scale=0.05, interval_width=0.4)
m.fit(df)

future = m.make_future_dataframe(periods=365)
# print future.tail()

forecast = m.predict(future)

print forecast.head()

# m.plot(forecast)
m.plot_components(forecast)

plot.show()
