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
def f(x):
    return x**2+10*np.sin(x)

if __name__ == '__main__':
    a=np.ones([3,3]);
    spio.savemat('file.mat', {'a':a})
    data=spio.loadmat("file.mat", struct_as_record=True);
    print(data['a'])
    arr=np.array([[1,2,3],[3,4,6],[5,7,8]]);
    print('arr:',linalg.det(arr));
    x=np.arange(-10,10,0.1)
    print("fmin:", optimize.fmin(f, -5) )
    plt.plot(x,f(x))
    plt.show()
    pass