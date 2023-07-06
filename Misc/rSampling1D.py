import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns

def p(x):
    return norm.pdf(x,loc=30,scale =10) + norm.pdf(x,loc=80,scale =20)#norm: 2

def q(x):
    return norm.pdf(x,loc=50,scale =30)

X=np.linspace(-50,150,1000)

k=max(p(X)/q(X))

def sample(size):
    xs=np.random.normal(50,30,size=size) #q(x)
    cs=np.random.uniform(0,1,size=size)
    mask = p(xs)/(k*q(xs))>cs
    return xs[mask]

samples = sample(10000)

plt.plot(X,p(X)/2,label="true PDF")
sns.distplot(samples)
plt.legend()
plt.title("rejection sampling 1D \nN={}".format(len(samples)))
plt.savefig("rSampling1D.pdf")