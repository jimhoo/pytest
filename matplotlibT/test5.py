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

#已知公式，求解
#例如y=1-x 和 3x+2y=5 求  x,y的解
#solve为求解



if __name__ == '__main__':
    x=Symbol("x")
    y=Symbol("y")
    a=solve([y+x-1,3*x+2*y-5],[x,y])
    solve_linear
    print(a)
    pass