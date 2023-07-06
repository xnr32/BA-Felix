import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det
import scipy.stats as st

import entropy_estimators as ee

N=1000
mu=[0,0]
dim=len(mu)
sigma=np.array([[1,0.9],[0.9,1]]) #[a,b],[c,d]

detsigma=(sigma[0,0]*sigma[1,1] - sigma[1,0]*sigma[0,1]) #1/(ab-cd)
sigma_i=np.array([[ sigma[1,1],sigma[0,1] ] , [ -sigma[1,0],sigma[0,0] ]])/detsigma #detsigma*[d,-b],[-c,a]

rv = st.multivariate_normal(mu, sigma)
sample=rv.rvs(N)


entest=ee.entropy(sample, k=3, base=np.exp(1))
entana=1/2*(dim + dim*log(2*np.pi) + log(detsigma))

iest=ee.mi(sample[:,0],sample[:,1],base=np.exp(1))
iana=-log(1-sigma[1,0]*sigma[1,0])/2

#%%

fig, (ax1) = plt.subplots(1,1,figsize=(10,7),sharex='col')
ax1.scatter(sample[:,0],sample[:,1],s=0.5,marker='o')
ax1.text(-2,2, "S estimated: {} \nS analytical: {}\nI estimated: {} \nI analytical: {}".format(entest,entana,iest,iana))
ax1.set_title("Bivariate Gaussian \n$\mu$={} $\sigma$=[{},{}] \nN={}".format(mu,sigma[0,:],sigma[1,:],N))
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
fig.savefig("entBiGAUSS.pdf")

