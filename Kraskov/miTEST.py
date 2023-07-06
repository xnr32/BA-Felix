import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det

import entropy_estimators as ee

mu1=0
sigma1=np.exp(0)

mu2=0
sigma2=np.exp(0)

X=np.array([[np.random.normal(mu1,sigma1)] for i in range(1000)])
Y=X
#Y=np.array([[np.random.normal(mu2,sigma2)] for i in range(1000)])

XY=np.c_[X,Y]

print(ee.entropy(XY, k=3))
print(ee.entropy(X, k=3))


#print(ee.mi(X,Y))

plt.scatter(X,Y,s=5)
plt.axis('equal')