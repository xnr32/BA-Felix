import numpy as np
import scipy.stats as st

bins=np.linspace(-3,3,10)
N=1000

data =np.c_[np.random.normal(size=N),np.random.normal(size=N)]
H=np.histogram2d(data[:,0],data[:,1],bins=bins)
P=H[0]/N

XY1=np.meshgrid(bins,bins)
XY=np.dstack((bins,bins))

rv = st.multivariate_normal([0,0], np.identity(2))
P_true=rv.pdf(XY)
