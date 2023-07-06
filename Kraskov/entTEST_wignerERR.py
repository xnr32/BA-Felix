import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det
import scipy.stats as st
import entropy_estimators as ee
#%% create sample

#create custom pdf
class custom(st.rv_continuous):
    def _pdf(self,x,y):
        return  2*np.exp(-4*np.pi**2 * x**2 - y**2) #MUST BE NORMALIZED!
    
cust_pdf = custom(name='cust_pdf')   

N=100

sample=cust_pdf.rvs(size=N)



#%%

sampleest=np.zeros((N,1))
j=0
for i in sample:
    sampleest[j,0]=sample[j]
    j+=1

entest=ee.entropy(sampleest, k=3)
entana=1.34273

print(
"estimated:",entest
)

print(
"analytical:",entana
)

#%%plot 
Y=np.zeros(N)
xplot=np.arange(-5,5,0.01)

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(7,10),sharex='col')
ax1.hist(sample,bins=21)
ax1.plot(xplot,(300*xplot**2*np.exp(-(xplot)**2)))
ax1.set_xlabel('X')
ax1.set_ylabel('N')
ax1.set_title("N={} \n Histogram".format(N))
ax2.scatter(sample,Y,s=0.5,marker='o')
ax2.set_xlabel('X')
ax2.set_title('Distribution')
ax2.text(-2,-0.02,"Entropy analytical: {} \nEntropy estimated:{}\nDifference:{}".format(entana,entest,abs(entana-entest)))
fig.savefig("entCUSTOM.pdf")

