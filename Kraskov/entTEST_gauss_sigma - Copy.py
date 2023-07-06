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
est3=np.zeros(len(sigma))
est6=np.zeros(len(sigma))
est9=np.zeros(len(sigma))
ana=np.zeros(len(sigma))

j=0
for j in range(len(sigma)):
    X=np.array([[np.random.normal(mu,sigma[j])] for i in range(N)])
    X3=np.round(X,3)
    X6=np.round(X,6)
    X9=np.round(X,9)
    est[j]=ee.entropy(X, k=3)
    ana[j]=log(sigma[j]*np.sqrt(2*np.pi*np.exp(1)),2)
    est3[j]=ee.entropy(X3, k=3)
    est6[j]=ee.entropy(X6, k=3)
    est9[j]=ee.entropy(X9, k=3)
    j+=1
    
# diff=[est[i]-ana[i] for i in range(len(sigma))]

#%%
# plt.scatter(sigma,diff, label="difference")

plt.scatter(sigma,est3,label="est. rounded 3 dig. ")
plt.scatter(sigma,est6,label="est. rounded 6 dig.")
plt.scatter(sigma,est9,label="est. rounded 9 dig.")
plt.scatter(sigma,ana,label="analytical")
plt.scatter(sigma,est,label="est. float64")
plt.legend()
plt.xscale("log")
plt.title("estimation vs analytical Entropy of a gaussian for different $\sigma$ \n N={}".format(N))
plt.xlabel("standard deviation $\sigma$")
plt.ylabel("Entropy S")
plt.savefig("entGAUSS_sigma_round.pdf")


