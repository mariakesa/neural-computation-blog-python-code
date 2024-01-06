import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap, utils
from scipy.stats import zscore

# spks is neurons by time
spks = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_clip.npy").T.astype("float32")
spks = zscore(spks, axis=1)

# fit rastermap
model = Rastermap(n_PCs=200, n_clusters=100, 
                  locality=0.75, time_lag_window=5).fit(spks)
y = model.embedding # neurons x 1
isort = model.isort

# bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

# plot
fig = plt.figure(figsize=(12,5))
ax = fig.add_subplot(111)
ax.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
plt.show()