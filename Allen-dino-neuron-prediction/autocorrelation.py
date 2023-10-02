import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf 

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
    #plt.plot(plot)
    plot_acf(dat[:,j])
    plt.show()

output_dir = '/media/maria/DATA/AllenData'

# Get data


class MakeTestTrain():
    def __init__(self, cre_line):
        self.cre_line = cre_line
        self.output_dir = '/media/maria/DATA/AllenData'
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(output_dir) / 'brain_observatory_manifest.json'))

