#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author = 'wyx'
@time = 2017/9/13 17:33
@annotation = ''
"""
import weakref


class LazyFunc(object):
    """描述器类：作用在类中需要lazy的对象方法上"""

    def __init__(self, func):
        """
        外部使用eg：
            class BuyCallMixin(object):
                @LazyFunc
                def buy_type_str(self):
                    return "call"

                @LazyFunc
                def expect_direction(self):
                    return 1.0
        """
        self.func = func
        self.cache = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        """描述器__get__，使用weakref.WeakKeyDictionary将以实例化的instance加入缓存"""
        if instance is None:
            return self
        try:
            return self.cache[instance]
        except KeyError:
            ret = self.func(instance)
            self.cache[instance] = ret
            return ret

    def __set__(self, instance, value):
        """描述器__set__，raise AttributeError，即禁止外部set值"""
        raise AttributeError("LazyFunc set value!!!")

    def __delete__(self, instance):
        """描述器___delete__从weakref.WeakKeyDictionary cache中删除instance"""
        del self.cache[instance]


class BuyCallMixin(object):
    """
        混入类，混入代表买涨，不完全是期权中buy call的概念，
        只代表看涨正向操作，即期望买入后交易目标价格上涨，上涨带来收益
    """

    @LazyFunc
    def buy_type_str(self):
        """用来区别买入类型unique 值为call"""
        return "call"

    @LazyFunc
    def expect_direction(self):
        """期望收益方向，1.0即正向期望"""
        return 1.0


class Demo(BuyCallMixin):
    pass


d = Demo()
print d
print d.buy_type_str()
