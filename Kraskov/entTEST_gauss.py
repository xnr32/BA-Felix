import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det
import scipy.stats as st
import entropy_estimators as ee

class custom(st.rv_continuous):
    def _pdf(self,x):
        return ( np.exp(-(x-2)**2) + np.exp(-(x+2)**2))/2/np.sqrt(np.pi) #MUST BE NORMALIZED!
    
cust_pdf = custom(name='cust_pdf')   

N=1000

# rv = st.norm
# sample=st.norm.rvs(size=N)

# rv = st.cust_pdf()
sample=cust_pdf.rvs(size=N)



#%%

# entest=ee.entropy(X, k=3)
# entana=log(sigma*np.sqrt(2*np.pi*np.exp(1)),2)

# print(
# "estimated:",entest
# )

# print(
# "analytical:",entana
# )


Y=np.zeros(N)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7,10),sharex='col')
ax1.hist(sample,bins=10)
ax1.set_xlabel('X')
ax1.set_ylabel('N')
ax1.set_title("N={} \n Histogram".format(N))
ax2.scatter(sample,Y,s=0.5,marker='o')
ax2.set_xlabel('X')
ax2.set_title('Distribution')
# ax2.text(-2,0.01,"1D Gaussian. $\mu=$ {}; $\sigma=${}".format(mu,sigma))
# ax2.text(-2,-0.02,"Entropy analytical: {} \nEntropy estimated:{}\nDifference:{}".format(entana,entest,abs(entana-entest)))
# fig.savefig("entGAUSS.pdf")

