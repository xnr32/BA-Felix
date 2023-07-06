import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det

import entropy_estimators as ee

N=1000
mu=0
sigma=[10**(-i) for i in range(25)]

est=np.zeros(len(sigma))
ana=np.zeros(len(sigma))

j=0
for j in range(len(sigma)):
    X=np.array([[np.random.normal(mu,sigma[j])] for i in range(N)])
    est[j]=ee.entropy(X, k=3)
    ana[j]=log(sigma[j]*np.sqrt(2*np.pi*np.exp(1)),2)
    j+=1
    
diff=[est[i]-ana[i] for i in range(len(sigma))]

plt.scatter(sigma,diff, label="difference")
plt.scatter(sigma,est,label="estimate")
plt.scatter(sigma,ana,label="analytical")
plt.legend()
plt.xscale("log")
plt.title("estimation vs analytical Entropy of a gaussian for different $\sigma$ \n N={}".format(N))
plt.xlabel("standard deviation $\sigma$")
plt.ylabel("Entropy S")
plt.savefig("entGAUSS_sigma.pdf")


