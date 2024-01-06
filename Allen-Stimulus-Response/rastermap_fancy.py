import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap, utils
from scipy.stats import zscore

# Load data
stimulus = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one.npy")
spks = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_clip.npy").T.astype("float32")
spks = zscore(spks, axis=1)

# Fit rastermap
model = Rastermap(n_PCs=200, n_clusters=100, locality=0.75, time_lag_window=5).fit(spks)
y = model.embedding
isort = model.isort

# Bin over neurons
X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)

# Plot
fig, axs = plt.subplots(ncols=6, nrows=2, figsize=(10, 5))
gs = axs[1, 0].get_gridspec()

# Remove the underlying axes
for ax in axs[1, :6]:
    ax.remove()

axbig = fig.add_subplot(gs[1:, :6])

# Plot 9 stimulus images side by side horizontally
for i, ax in enumerate(axs[0, :]):
    ax.imshow(stimulus[i*100], cmap="gray", aspect="auto")
    ax.axis('off')

# Plot X_embedding in the bottom subplot
axbig.imshow(X_embedding, vmin=0, vmax=1.5, cmap="gray_r", aspect="auto")
axbig.axis('off')

# Draw vertical lines indicating the center of each image on X_embedding
for i in range(0, 6):
    axbig.axvline(x=i * np.floor(900/5), color='red', linestyle='--')

fig.tight_layout()
plt.show()
