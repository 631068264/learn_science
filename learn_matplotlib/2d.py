#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/17 09:25
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

# sns.set()
u = np.linspace(-2, 2, 65)
v = np.linspace(-1, 1, 33)
# print u, v
X, Y = np.meshgrid(u, v)
# print X, Y
Z = X ** 2 / 25 + Y ** 2 / 4

print Z
# plt.set_cmap('gray')

# plt.pcolor(Z)
# plt.colorbar()

plt.contour(Z)
plt.show()
