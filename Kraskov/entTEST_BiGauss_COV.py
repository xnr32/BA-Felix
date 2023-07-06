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


brng=np.arange(0,1,0.01)
entest=np.zeros(len(brng))
entana=np.zeros(len(brng))
iest=np.zeros(len(brng))
iana=np.zeros(len(brng))

j=0
for i in brng:
#cov-matrix
    sigma=np.array([[1,brng[j]],[brng[j],1]]) #[a,b],[c,d]
    detsigma=(sigma[0,0]*sigma[1,1] - sigma[1,0]*sigma[0,1]) #1/(ab-cd)
    sigma_i=np.array([[ sigma[1,1],sigma[0,1] ] , [ -sigma[1,0],sigma[0,0] ]])/detsigma #detsigma*[d,-b],[-c,a]
#create sample
    rv = st.multivariate_normal(mu, sigma)
    sample=rv.rvs(N)
#write entropies
    entest[j]=ee.entropy(sample, k=3, base=np.exp(1))
    entana[j]=1/2*(dim + dim*log(2*np.pi) + log(detsigma))
#write MI    
    iest[j]=ee.mi(sample[:,0],sample[:,1],base=np.exp(1))
    iana[j]=-log(1-sigma[1,0]*sigma[1,0])/2
    
    j+=1

#%%
fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7,10),sharex='col')
ax1.scatter(brng,entest,s=0.5,marker='o',label="estimated entropy")
ax1.scatter(brng,entana,s=0.5,marker='o',label="analytical entropy")
# ax1.text(-2,2, "S estimated: {} \nS analytical: {}\nI estimated: {} \nI analytical: {}".format(entest,entana,iest,iana))
ax1.set_title("Bivariate Gaussian: variable COV R \n$\mu$={} $\sigma$=[[1,R][R,1]] \nN={}".format(mu,N))
ax1.set_xlabel('Covariance R')
ax1.set_ylabel('joint entropy H(X1,X2)')
ax1.legend()

ax2.scatter(brng,iest,s=0.5,marker='o',label="estimated MI")
ax2.scatter(brng,iana,s=0.5,marker='o',label="analytical MI")
ax2.set_xlabel('Covariance R')
ax2.set_ylabel('mutual Information I(X1,X2)')
ax2.legend()
fig.savefig("entBiGAUSS_COV.pdf")

