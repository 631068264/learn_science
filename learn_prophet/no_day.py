#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/22 11:31
@annotation = ''
"""

import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plot

df = pd.read_csv('example_retail_sales.csv')
# df['y'] = np.log(df['y'])
print df.head()
print df.tail()

m = Prophet()
m.fit(df)

future = m.make_future_dataframe(periods=120, freq='M')
# print future.tail()

forecast = m.predict(future)

print forecast.head()

# m.plot(forecast)
m.plot_components(forecast)

plot.show()
