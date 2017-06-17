#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/6/17 12:23
@annotation = ''
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])

z = np.polyfit(x, y, 30)
print z
p = np.poly1d(z)
print p
xp = np.linspace(-2, 6, 100)

plt.plot(x, y, '.', xp, p(xp), "-")

plt.show()
