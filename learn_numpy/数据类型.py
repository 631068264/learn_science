#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/1/21 16:16
@annotation = '' 
"""
import datetime
from decimal import Decimal

import pandas as pd

"""
数据类型转换
数字代表在内存占用位数 可以
bool
inti
int8
int16
int32
int64
uint8
uint16
uint32
uint64
float16
float32 float64或float

complex64 complex128或complex

用一位存储的布尔类型(值为TRUE或FALSE)
 由所在平台决定其精度的整数(一般为int32或int64)
整数，范围为128至127 7 整数，范围为32 768至32 767
整数，范围为231至231 1
整数，范围为263至263 1
无符号整数，范围为0至255
无符号整数，范围为0至65 535
无符号整数，范围为0至2321
无符号整数，范围为0至2641 半精度浮点数(16位):其中用1位表示正负号，5位表示指数，10位表示尾数 单精度浮点数(32位):其中用1位表示正负号，8位表示指数，23位表示尾数 双精度浮点数(64位):其中用1位表示正负号，11位表示指数，52位表示尾数 复数，分别用两个32位浮点数表示实部和虚部 复数，分别用两个64位浮点数表示实部和虚部
"""
import numpy as np
from numpy import dtype

# print np.arange(10)

# 浮点可以转复数
m = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.float16)
if False:
    print(m.dtype)
    print(m)
    print(np.float16(10))

"""
自定义数据类型
大端序是将最高位字节存储在最低的内存地址处，用>表示;与之相反，小端序 是将最低位字节存储在最低的内存地址处，用<表示
"""
t = dtype([('name', np.str, 40), ('numitems', np.int32), ('price', np.float32)])
if False:
    print(t['name'])
    print(t['name'].char)
    print(t)
    itemz = np.array([('Meaning of life DVD', 42.9954, 3.14), ('Butter', 13, 2.72)], dtype=t)
    print(itemz[0])

rows = [{'date': datetime.datetime(2017, 3, 21, 13, 27, 21), 'tid': 648605515L, 'price': Decimal('1080.04000000'),
         'type': 1, 'amount': Decimal('10.00000000')},
        {'date': datetime.datetime(2017, 3, 21, 13, 27, 22), 'tid': 648605549L, 'price': Decimal('1081.55000000'),
         'type': 1, 'amount': Decimal('16.00000000')},
        {'date': datetime.datetime(2017, 3, 21, 13, 27, 23), 'tid': 648605547L, 'price': Decimal('1081.33000000'),
         'type': 1, 'amount': Decimal('20.00000000')},
        {'date': datetime.datetime(2017, 3, 21, 13, 27, 24), 'tid': 648605545L, 'price': Decimal('1081.30000000'),
         'type': 1, 'amount': Decimal('16.00000000')},
        {'date': datetime.datetime(2017, 3, 21, 13, 27, 26), 'tid': 648605543L, 'price': Decimal('1081.29000000'),
         'type': 1, 'amount': Decimal('20.00000000')},
        # {'date': datetime.datetime(2017, 3, 22, 13, 27, 20), 'tid': 648605541L, 'price': Decimal('1080.46000000'),
        #  'type': 1, 'amount': Decimal('26.00000000')},
        # {'date': datetime.datetime(2017, 3, 22, 13, 27, 20), 'tid': 648605517L, 'price': Decimal('1080.04000000'),
        #  'type': 1, 'amount': Decimal('8.00000000')},
        # {'date': datetime.datetime(2017, 3, 23, 13, 27, 22), 'tid': 648605601L, 'price': Decimal('1079.69000000'),
        #  'type': -1, 'amount': Decimal('70.00000000')},
        # {'date': datetime.datetime(2017, 3, 24, 13, 27, 25), 'tid': 648605686L, 'price': Decimal('1079.72000000'),
        #  'type': -1, 'amount': Decimal('4.00000000')}
        ]

df = pd.DataFrame(rows)
df[["amount", "price"]] = df[["amount", "price"]].astype(
    "float64")
df["date"] = df["date"].astype(pd.Timestamp)

print df.dtypes
fileds = df.columns.values
result = np.empty(shape=(len(df)), dtype=np.dtype([(field, df[field].dtype) for field in fileds]))

for field in fileds:
    result[field] = df[field][:len(df)]

# print result
# print type(result)
# print result["date"]
# print type(result["date"])
print len(result['date'])
print result[0,0]
pos = result["date"].searchsorted(pd.Timestamp(datetime.datetime(2017, 3, 21, 13, 27, 24)))
print pos
