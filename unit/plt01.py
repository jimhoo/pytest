'''
Created on 2018年6月6日

@author: Administrator
'''
from sympy import *
import matplotlib.pyplot as plt
import numpy as np
from math import *
from sympy.core.function import diff

if __name__ == '__main__':
    x=Symbol("x");
    y=Symbol("y");

#     从[-1,1]中等距去50个数作为x的取值
#     x=np.linspace(-1,1,50);
#     y=2*x+1;
#     plt.xlabel("x");
#     plt.ylabel("y");
#     plt.plot(x,y);
#     plt.show();
#     
    x=np.linspace(-np.pi,np.pi,256,endpoint=True);
    y=np.sin(x);
    #z=e^x
    z=np.exp(x); 
    #a=2^x
    a=np.power(2,x);
    b=3**x;
    c=x**3;
    j=Symbol("j");
    cc=j**3;
    fb=diff(cc,j,1);
    fb0=fb.subs(j,2);
    print(fb);
    print(fb0);
    
#     plt.plot(x,y);
#     plt.plot(x,z);
#     plt.plot(x,a);
#     plt.plot(x,b);
#     plt.plot(x,c);
    plt.show();

    pass