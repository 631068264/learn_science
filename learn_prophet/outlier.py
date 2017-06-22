#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/22 11:20
@annotation = ''
"""
import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plot

df = pd.read_csv('example_wp_R_outliers1.csv')
df['y'] = np.log(df['y'])
print df.head()
print df.tail()

m = Prophet()
# df.loc[(df['ds'] > '2010-01-01') & (df['ds'] < '2011-01-01'), 'y'] = None
m.fit(df)
"""
解决过度离散值
The best way to handle outliers is to remove them - Prophet has no problem with missing data.
If you set their values to NA in the history but leave the dates in future,
then Prophet will give you a prediction for their values.
"""
future = m.make_future_dataframe(periods=365)
# print future.tail()

forecast = m.predict(future)

print forecast.head()

# m.plot(forecast)
m.plot_components(forecast)

plot.show()
