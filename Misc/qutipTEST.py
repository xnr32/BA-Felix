from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from qutip import *

alpha=1

# rho_fock=fock_dm(3,2)
rho_coh=coherent_dm(3,alpha)

X=np.linspace(-5,5,100)

Q_coh=qfunc(rho_coh,X,X)
# Q_fock=qfunc(rho_fock,X,X)

# measure(rho_coh,)
rho_coh






# X, Y = np.meshgrid(X, X)
# fig=plt.figure()
# ax=fig.add_subplot(projection='3d')
# ax.plot_wireframe(X,Y,Q_coh, rstride=3, cstride=10)
# ax.set_xlabel('X')
# ax.set_ylabel('P')
# ax.set_zlabel('Q(X,P)')
# ax.set_title("Husimi Distribution for coherent state of Fock-0 State \nalpha= {}".format(alpha))
 