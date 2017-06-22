#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/22 10:04
@annotation = ''
"""
import numpy as np
import pandas as pd
from fbprophet import Prophet
from matplotlib import pyplot as plot

"""
Prophet allows you to make forecasts using a logistic growth trend model, with a specified carrying capacity.


"""
df = pd.read_csv('example_wp_R.csv')
df['y'] = np.log(df['y'])
df['cap'] = 8.5
m = Prophet(growth='logistic')
m.fit(df)

future = m.make_future_dataframe(periods=1826)
future['cap'] = 8.5
fcst = m.predict(future)
m.plot_components(fcst)

plot.show()
