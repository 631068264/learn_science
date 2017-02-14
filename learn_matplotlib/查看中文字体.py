#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/31 12:19
@annotation = '' 
"""
import subprocess

import matplotlib
from matplotlib.font_manager import FontManager
from pylab import mpl

print matplotlib.matplotlib_fname()
fm = FontManager()
mat_fonts = set(
    f.name for f in fm.ttflist if f.name.startswith("Sim") or f.name.startswith("Apple") or f.name.startswith("Li"))
print mat_fonts
print mpl.rcParams['font.sans-serif']
print mpl.rcParams['font.family']

# output = subprocess.check_output(
#     'fc-list :lang=zh -f "%{family}\n"', shell=True)
# # print '*' * 10, '系统可用的中文字体', '*' * 10
# # print output
# zh_fonts = set(f.split(',', 1)[0] for f in output.split('\n'))
# available = mat_fonts & zh_fonts
#
# print '*' * 10, '可用的字体', '*' * 10
# for f in available:
#     print f
