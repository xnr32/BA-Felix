import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det
import scipy.stats as st

import entropy_estimators as ee

Nrng=np.logspace(1,4,num=4)
Nrng=Nrng.astype(int)
mu=[0,0]
dim=len(mu)


brng=np.arange(0,1,0.01)
iest=np.zeros((len(brng),len(Nrng)))
iana=np.zeros((len(brng),len(Nrng)))

k=0
for l in Nrng:
    j=0
    for i in brng:
    #cov-matrix
        sigma=np.array([[1,brng[j]],[brng[j],1]]) #[a,b],[c,d]
        detsigma=1/(sigma[0,0]*sigma[1,1] - sigma[1,0]*sigma[0,1]) #1/(ab-cd)
        sigma_i=detsigma*np.array([[ sigma[1,1],sigma[0,1] ] , [ -sigma[1,0],sigma[0,0] ]]) #detsigma*[d,-b],[-c,a]
    #create sample
        rv = st.multivariate_normal(mu, sigma)
        sample=rv.rvs(Nrng[k])
    # #write entropies
    #     entest[j]=ee.entropy(sample, k=3, base=np.exp(1))
    #     entana[j]=1/2*(dim + dim*log(2*np.pi) + log(detsigma))
    #write MI    
        iest[j,k]=ee.mi(sample[:,0],sample[:,1],base=np.exp(1))
        iana[j,k]=-log(1-sigma[1,0]*sigma[1,0])/2
        
        j+=1
       
    k+=1

#%%

fig, axes = plt.subplots(4,1,figsize=(10,15),sharex='col')

j=0
for i in axes:
    i.scatter(brng,iest[:,j],s=0.5,marker='o',label="estimated MI")
    i.scatter(brng,iana[:,j],s=0.5,marker='o',label="analytical MI")
    i.set_title("N={}".format(Nrng[j]))
    i.legend()
    i.set_ylabel('I(X1,X2)')
    
    j+=1
    
i.set_xlabel('Covariance R')

fig.savefig("entBiGAUSS_COV_N.pdf")

