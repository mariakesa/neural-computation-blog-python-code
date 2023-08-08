import numpy as np
import matplotlib.pyplot as plt

X = np.load('results_X.npy')
print(X.shape)
Y = np.load('results_Y.npy')
print(Y.shape)
print(np.corrcoef(X.T, Y.T))
plt.scatter(X[:, 3], Y[:, 3])
plt.show()
