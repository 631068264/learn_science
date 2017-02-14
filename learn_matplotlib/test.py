#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/31 11:28
@annotation = '' 
"""
# from pylab import mpl
# # print mpl.rcParams
# # mpl.rcParams['font.sans-serif'] = u'SimHei'  # 指定默认字体
# # mpl.rcParams['font.family'] = u'SimHei'  # 指定默认字体
# # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

import matplotlib.pyplot as plt
from numpy.random import randn

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)

ax1.plot(randn(50).cumsum(), "k--", color="g")
ax1.set_title(u"我是title")
ax1.set_ylabel(u"y", rotation=45, )

ax2.plot(randn(50).cumsum(), color="k", linestyle="dashed", marker="o")
ax2.text(20, 3, "Hello world", fontsize="10", color="b")

ax3.plot(randn(100).cumsum(), color="r", drawstyle="steps-post", label="k_label")
ax3.plot(randn(100).cumsum(), color="g", drawstyle="steps-post", label="g_label")
# 图例
ax3.legend(loc="best")
ax3.set_xticks([0, 25, 50, 75, 100])
ax3.set_xticklabels(list("ABCDE"), fontsize="small")
ax3.set_xlabel(u"x")

# 保存
plt.savefig("figure.png", bbox_inches="tight")

# 流
from io import StringIO

buffer = StringIO()
plt.savefig(buffer)
plot_data = buffer.getvalue()

plt.show()
