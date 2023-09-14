import numpy as np
import matplotlib.pyplot as plt

path='/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_movie_one.npy'
path='/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_natural_scenes.npy'

dat=np.load(path)

print(dat.shape)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    result=result[result.size // 2:]
    return result/float(result.max())

for j in range(0,768):
    plot=autocorr(dat[:,j])
    plt.plot(plot)

plt.show()

