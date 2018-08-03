'''
Created on 2018年6月1日

@author: Administrator
'''
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
n=10
p=0.3
k=np.arange(0,21)
binomial=binom.pmf(k,n,p)
print(binomial)

plt.plot(k,binomial,'-')
plt.title("Binomail:n=%i,p=%0.2f" %(n,p),fontsize=15)
plt.xlabel("Number of successes")
plt.ylabel("Probability of sucesses",fontsize=15)
plt.show()
