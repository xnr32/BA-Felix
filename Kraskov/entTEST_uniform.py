import random
from math import log, pi
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nr
from numpy.linalg import det

import entropy_estimators as ee

P_width=2

X=np.array([[P_width*random.random()] for i in range(1000)])

print(
"estimated:",ee.entropy(X, k=3)
)

print(
"analytical:",-log(1/P_width,2)      
      )

Y=np.zeros(len(X))


fig, (ax1, ax2) = plt.subplots(2,figsize=(7,10))
ax1.hist(X,bins=10)
ax2.scatter(X,Y,s=1,marker='o')

