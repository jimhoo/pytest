'''
Created on 2018年7月1日

@author: Administrator
'''
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy import fftpack as fft
from scipy import io as spio
from scipy import linalg 
from scipy import optimize
from scipy import stats

from scipy.special import perm
from scipy.special import comb
from scipy.special import factorial 
from sympy import *
import math
from sympy.core.symbol import Symbol
from pip._vendor.distlib.compat import raw_input

# rvs：对随机变量进行随机取值，可以通过size参数指定输出的数组的大小。
# pdf：随机变量的概率密度函数。
# cdf：随机变量的累积分布函数，它是概率密度函数的积分。
# sf：随机变量的生存函数，它的值是1-cdf(t)。
# ppf：累积分布函数的反函数。
# stat：计算随机变量的期望值和方差。
# fit：对一组随机取样进行拟合，找出最适合取样数据的概率密度函数的系数。



if __name__ == '__main__':
    x=[1,2,3,4,5,6]
    y=[2,5,7,9,12,13]
    a=Symbol("a")
    f=-0.1*a**4-0.15*a**3-0.5*a**2-0.25*a+1.2
    f1=diff(f,a,1)
    f2=diff(f,a,2)
    f3=diff(f,a,3)
    f4=diff(f,a,4)
    print(f1,f2,f3,f4)
    g0=f.subs(a,0)
    g1=f1.subs(a,0)
    g2=f2.subs(a,0)
    g3=f3.subs(a,0)
    g4=f4.subs(a,0)
    a=int(raw_input("请输入改变后的a值:"))
    g=g0/math.factorial(0)*a**0+g1/math.factorial(1)*a**1+g2/math.factorial(2)*a**2+g3/math.factorial(3)*a**3+g4/math.factorial(4)*a**4
    print("f",f)
    print("g",g)
    print("排列A5(2):",perm(5, 2))
    print("组合C5(2):",comb(5, 2))
    print("阶乘5!:",factorial(5));
    plt.plot(x,y);
    plt.show();
    pass