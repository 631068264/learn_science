#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/31 17:22
@annotation = '' 
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import randn

s1 = pd.Series(randn(10).cumsum(), index=np.arange(0, 100, 10))
s1.plot()
s1.plot(kind="bar", color="g")

df = pd.DataFrame(randn(10, 4).cumsum(0), index=np.arange(0, 100, 10), columns=list("ABCD"))
df.plot()
df.plot(kind="barh", stacked=True)
df.plot(kind="kde")
plt.show()
