import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det

import entropy_estimators as ee


mu=0
sigma=[10**(-i) for i in range(25)]

est=np.zeros(len(sigma))
ana=np.zeros(len(sigma))

j=0
for j in range(len(sigma)):
    X=np.array([[np.random.normal(mu,sigma[j])] for i in range(100)])
    est[j]=ee.entropy(X, k=3)
    ana[j]=log(sigma[j]*np.sqrt(2*np.pi*np.exp(1)),2)
    j+=1
    
diff=[est[i]-ana[i] for i in range(len(sigma))]

plt.scatter(sigma,diff, label="difference")
plt.scatter(sigma,est,label="estimate")
plt.scatter(sigma,ana,label="analytical")
plt.legend()
plt.xscale("log")

# print(
# "estimated:",ee.entropy(X, k=3)
# )

# print(
# "analytical:",log(sigma*np.sqrt(2*np.pi*np.exp(1)),2)
# )


#Y=np.zeros(len(X))


# fig, (ax1, ax2) = plt.subplots(2,figsize=(7,10))
# ax1.hist(X,bins=10)
# ax2.scatter(X,Y,s=1,marker='o')

