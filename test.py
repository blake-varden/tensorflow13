from data import NumericalDataProvider


import numpy as np


X = np.ones([10, 5])
y = np.ones([10, 1]) * 5

data = [ [X[i], y[i]] for i in range(len(X))]

provider = NumericalDataProvider(5)