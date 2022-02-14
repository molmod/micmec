import numpy as np

N = 10
masses = np.arange(N)
pos = np.ones((N, 3))
print(masses.shape)
masses = masses.reshape(-1, 1)
print(masses.shape)
