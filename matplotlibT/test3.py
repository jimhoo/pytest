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

# rvs：对随机变量进行随机取值，可以通过size参数指定输出的数组的大小。
# pdf：随机变量的概率密度函数。
# cdf：随机变量的累积分布函数，它是概率密度函数的积分。
# sf：随机变量的生存函数，它的值是1-cdf(t)。
# ppf：累积分布函数的反函数。
# stat：计算随机变量的期望值和方差。
# fit：对一组随机取样进行拟合，找出最适合取样数据的概率密度函数的系数。

def f(x):
    return x**2+10*np.sin(x)

if __name__ == '__main__':
    a=np.random.normal(size=1000)
    loc,std=stats.norm.fit(a)
    print("mean:",np.mean(a))
    print("u,std",loc,std)
    X = stats.norm(loc=1.0, scale=2.0)
    print("X.stats:",X.stats());
    x=X.rvs(size=10000);
    print("u,std:",stats.norm.fit(x));
#     plt.plot(x,X.pdf(x));
    t=np.arange(-10,10,0.1);
    plt.plot(t,X.pdf(t));
#histogram()对数组x进行直方图统计，它将数组x的取值范围分为100个区间，并统计x中的每个值落入各个区间中的次数     
    p, t2 = np.histogram(x, bins=100, normed=True)
    t2 = (t2[:-1] + t2[1:])/2
    plt.plot(t2, p) # 绘制统计所得到的概率密度
#     plt.plot(t,X.cdf(t));
    plt.show();
    pass