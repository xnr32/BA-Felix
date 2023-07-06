# import gmm_mi.mi as mi
from gmm_mi.mi import EstimateMI
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


N=1000
mu=[0,0]
dim=len(mu)
sigma=np.array([[1,0.9],[0.9,1]]) #[a,b],[c,d]

mi_estimator = EstimateMI()

rv = st.multivariate_normal(mu, sigma)
sample=rv.rvs(N)
mi_estimator = EstimateMI() 
iest = mi_estimator.fit_estimate(sample)
iana=-np.log(1-sigma[1,0]*sigma[1,0])/2
    
