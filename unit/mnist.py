
'''
Created on 2018��5��29��
@author: Administrator
'''
# from sympy import *
# init_printing(use_unicode=True)
# x,y = symbols('x y')  #用符号代表变量，多个变量可以空格，可以逗号隔开。
# expr = x + 2*y
# expanded_expr = expand(x*expr) #expand 展开
# factor(expanded_expr) # factor合并
# diff(sin(x)*exp(x), x) #对x求导
# integrate(exp(x)*sin(x) + exp(x)*cos(x), x) #对x求不定积分
# integrate(sin(x**2), (x, -oo, oo)) #对x求定积分
# limit(sin(x)/x, x, 0) #x趋于0的极限
# solve(x**2 - 2,x) #求方程的解
# Matrix([[1, 2], [2, 2]]).eigenvals() #求矩阵特征值
# latex(Integral(cos(x)**2, (x, 0, pi))) #查看latex格式
# expr = x + 1
# expr.subs(x,y) # y+1 替代方法subs
# Eq(x + 1, 4) # 建立方程 x+1 = 4 
# a = (x + 1)**2
# b = x**2 + 2*x + 1
# simplify(a-b) # 0 用这个方法验证两个方程是否相等
# a.equals(b) #或者equals方法验证
# Rational(1, 2) 分数表示
# expr = expr = sqrt(8)
# expr.evalf(chop =True) #小数表示，去掉 Round-off error 
# expr = x*y + x - 3 + 2*x**2 - z*x**2 + x**3
# collect(expr, x) #collect，对x合并同类项
# cancel((x**2 + 2*x + 1)/(x**2 + x)) #cancel函数约分
# expr = (4*x**3 + 21*x**2 + 10*x + 12)/(x**4 + 5*x**3 + 5*x**2 + 4*x) 
# apart(expr) #拆分
# asin(1) #反三角函数
# trigsimp(sin(x)**4 - 2*cos(x)**2*sin(x)**2 + cos(x)**4) #trigsimp简化三角函数
# expand_trig(sin(x + y)) #分解三角函数
# powsimp(x**a*x**b) #简化指数运算
# 解方程： 
# 7）solveset()可以对方程进行符号求解，它的第一个参数是表示方程的表达式，其后的参数是表示方程中未知量的符号。
# 求导： 
# 8）Derivative是表示导函数的类，它的第一个参数是需要进行求导的数学函数，第二个参数是求导的自变量，注意：Derivative所得到的是一个导函数，它不会进行求导运算。t=Derivative(sin(x),x) 
# 9）调用doit()方法求出导函数。t.doit() 
# 10）也可以直接使用diff()函数或表达式的diff()方法来计算导函数。diff(sin(2*x),x)或sin(2*x).diff(x)。 
# 11）添加更多的符号参数可以表示高阶导函数，如：Derivative(f(x),x,x,x)，也可以表示多个变量的导函数，如：Derivative(f(x,y),x,2,y,3)
# 求微分： 
# 12）dsolve()可以对微分方程进行符号求解，它的第一个参数是一个带未知函数的表达式，第二个参数是需要进行求解的未知函数。它在解微分方程中可以传递hint参数，指定微分方程的解法，若设置为“best”则放dsolve()尝试所有的方法并返回最简单的解。 
# 求积分： 
# 13）integrate(f,x):计算不定积分；integrate(f,(x,a,b)):计算定积分；integrate(f,x,y):计算双重不定积分；integrate(f,(x,a,b),(y,a,b)):计算双重定积分。

import numpy as np
from unittest.test.testmock.testpatch import function
import math as mt

from sympy.core.symbol import Symbol
from sympy.core.function import diff
from sympy import *

if __name__ == '__main__':
    
    
    
    #SymPy有三个内建的数值类型：实数，有理数和整数。
    a=Rational(1/2);
    print(a);
    a=Rational(1,3);
    print(a)
    
    #代数式的合并together(expr,x)
    x=Symbol("x")
    y=Symbol("y")
    z=Symbol("z")
    print( together( (1/x+1/y+1/z), 1) );
    
    #阶乘
    y=factorial(x);
    print( y );
    print( y.subs(x,3) );
    
    
    n=2;
    a=np.e**n;
    b=mt.pow(2, 3);
    c=mt.exp(2);
    print("a: %10.8f" %a);
    print("b:%d" %b);
    print("c: %10.8f" %c);
    print(" test ");
    
    x=Symbol("x")
    f=-0.1*x**4-0.15*x**3-0.5*x**2-0.25*x+1.2
    g=(1+1/x)**x
    f1=diff(f,x,1);
    f2=diff(f,x,2);
    f3=diff(f,x,3);
    f4=diff(f,x,4);
    g1=diff(g,x,1)
    print (f1)
    print (f2)
    print (f3)
    print (f4)
    print (g1)
    #传入x=0求出各阶导函数的具体数值  ,变量替换subs函数
    g0 = f.subs(x,0)
    g1 = f1.subs(x,0)
    g2 = f2.subs(x,0)
    g3 = f3.subs(x,0)
    g4 = f4.subs(x,0)
    #当x改变时传入x的值
    x=int(input("请输入改变后x的值："))
    #将x与导函数的值带入泰勒公式中求出结果
    g = g0/mt.factorial(0) * x**0 + g1/mt.factorial(1) * x**1 + g2/mt.factorial(2) * x**2 + g3/mt.factorial(3) * x**3 + g4/mt.factorial(4) * x**4
    print(g)
    
    n=Symbol("n");
    s=(n+3)/((n+2))**n;
    #求s在x=1点的极限
    print(limit(s,x,1));
    #极限
    x=Symbol("x");
    y=Symbol("y");
    print( limit(sin(x)/x,x,0 ) );
    print(limit(sin(x) , x, 0) );
    #微分 ，    diff(func,var,n) n为高阶
    print( diff(sin(x),x) );
    print( diff(sin(x-1) , x, 2 ) );
    #偏微积分
    print( diff( (3*x*y+2*y-x), x,1) );
    #积分
    print( integrate(6*x**5,x) );
    #定积分
    print( integrate(x**3, (x,-1,3) ) );
    #计算广义积分
    print( integrate(exp(x), (x,0,oo) )  );
    pass