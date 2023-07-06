import numpy as np
from sklearn.neighbors import KDTree
rng = np.random.default_rng()
X = rng.random((10, 3))  # 10 points in 3 dimensions
tree = KDTree(X, leaf_size=2)              
dist, ind = tree.query(X[:1], k=3)
test=tree.query(X, k=3) 