# import gmm_mi.mi as mi
from gmm_mi.mi import EstimateMI
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


N=1000
mu=[0,0]
dim=len(mu)
sigma=np.array([[1,0],[0,1]]) #[a,b],[c,d]

mi_estimator = EstimateMI()

brng=np.arange(0,1,0.1)
# entest=np.zeros(len(brng))
# entana=np.zeros(len(brng))
iest=np.zeros(len(brng))
iesterr=np.zeros(len(brng))
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
# #write entropies
#     entest[j]=ee.entropy(sample, k=3, base=np.exp(1))
#     entana[j]=1/2*(dim + dim*log(2*np.pi) + log(detsigma))
#write MI    
    MI=mi_estimator.fit_estimate(sample)
    iest[j]= MI[0]
    iesterr[j] = MI[1]
    iana[j]=-np.log(1-sigma[1,0]*sigma[1,0])/2
    
    j+=1

#%%
fig, (ax2) = plt.subplots(1,1,figsize=(10,7),sharex='col')

ax2.errorbar(brng,iest,yerr=iesterr,label="estimated MI")
ax2.scatter(brng,iana,s=0.5,marker='o',label="analytical MI")
ax2.set_xlabel('Covariance R')
ax2.set_ylabel('mutual Information I(X1,X2)')
ax2.legend()
# fig.savefig("entBiGAUSS_COV.pdf")
