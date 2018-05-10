#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2018/5/6 12:17
@annotation = ''
"""
import scipy as sp

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')
print(data.shape)

x = data[:, 0]
y = data[:, 1]
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]
print(x, y)


def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)


fp1, res, rank, sv, rcond = sp.polyfit(x, y, deg=2, full=True)
print("Model parameters: %s" % fp1)
print("Error of the model:", res)
# deg =1 f(x) = kx+b fp1 = (k,b)
f1 = sp.poly1d(fp1)
print(f1)
f1e = error(f1, x, y)
print(f1e)

from scipy.optimize import fsolve

print(f1 - 1000)
print(fsolve(f1 - 1000, 800))

f2 = f1 - 1000
print(f2(249.90437846))

