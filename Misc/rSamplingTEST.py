import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import numpy as np
import scipy.stats as st
import seaborn as sns

def det(sigma):
    return 1/(sigma[0,0]*sigma[1,1] - sigma[1,0]*sigma[0,1]) #1/(ab-cd)

def invert(sigma):
    det(sigma)*np.array([[ sigma[1,1],sigma[0,1] ] , [ -sigma[1,0],sigma[0,0] ]]) #detsigma*[d,-b],[-c,a]

def p(x,y):
    return x**2 * np.exp(-x**2 -y**2)*2/np.pi

# def gauss(x,sigma,mu):
#     return np.exp(-(x-mu)@np.linalg.inv(sigma)@(np.transpose(x-mu))/2)/np.sqrt(2*np.pi * det(sigma))

sigma=np.array([[2,0],[0,2]])
mu=[0,0]

Xlin=np.linspace(-5,5,100)
X, Y = np.meshgrid(Xlin, Xlin,)
pos=np.empty(X.shape+(2,))
pos[:,:,0]=X
pos[:,:,1]=Y
rv = st.multivariate_normal(mu, sigma)

k=np.max(p(X,Y)/rv.pdf(pos))


def sample(size):
    xs=rv.rvs(size) #q(x)
    #cs=np.c_[np.random.uniform(0,1,size=size),np.random.uniform(0,1,size=size)]
    cs=np.random.uniform(0,1,size=size)
    mask = p(xs[:,0],xs[:,1])/(k*rv.pdf(xs))>cs
    return xs[mask]

samples = sample(10000)
#%%
fig=plt.figure()
ax=fig.add_subplot(2,1,1,projection='3d')
surf=ax.plot_wireframe(X,Y,p(X,Y),rstride=5,cstride=5,antialiased=False,linewidth=.1,alpha=0.5)
# surf=ax.plot_surface(X,Y,k*rv.pdf(pos),alpha=.2,antialiased=False)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("P(X,Y)")
ax.set_title("2D Rejection Sampling")

ax=fig.add_subplot(2,1,2)
ax.scatter(samples[:,0],samples[:,1],s=0.5,marker='o')
ax.set_xlabel("X")
ax.set_ylabel("Y")

