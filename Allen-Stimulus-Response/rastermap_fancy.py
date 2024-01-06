import numpy as np
import matplotlib.pyplot as plt
from rastermap import Rastermap, utils
from scipy.stats import zscore

# Load data
stimulus = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one.npy")
spks_clip = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_clip.npy").T.astype("float32")
spks_clip = zscore(spks_clip, axis=1)

spks_dino = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_dino.npy").T.astype("float32")
spks_dino = zscore(spks_dino, axis=1)

spks_vit = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_vit.npy").T.astype("float32")
spks_vit = zscore(spks_vit, axis=1)

def make_rastermap(spks):
    # Fit rastermap
    model = Rastermap(n_PCs=200, n_clusters=100, locality=0.4, time_lag_window=5).fit(spks)
    y = model.embedding
    isort = model.isort

    srtd = spks[isort]

    # Bin over neurons
    X_embedding = zscore(utils.bin1d(spks[isort], bin_size=25, axis=0), axis=1)
    return X_embedding, srtd

# Comment out the following lines to avoid overwriting the variables before plotting
#emb_clip, srtd_clip = make_rastermap(spks_clip)
#emb_dino, srtd_dino = make_rastermap(spks_dino)
#emb_vit, srtd_vit = make_rastermap(spks_vit)

#np.save('emb_clip.npy', emb_clip)
#np.save('emb_dino.npy', emb_dino)
#np.save('emb_vit.npy', emb_vit)

#np.save('srtd_clip.npy', srtd_clip)
#np.save('srtd_dino.npy', srtd_dino)
#np.save('srtd_vit.npy', srtd_vit)

# Load saved variables
emb_clip = np.load('emb_clip.npy')
emb_dino = np.load('emb_dino.npy')
emb_vit = np.load('emb_vit.npy')

srtd_clip = np.load('srtd_clip.npy').T
srtd_dino = np.load('srtd_dino.npy').T
srtd_vit = np.load('srtd_vit.npy').T

#plt.plot(srtd_vit[0])
#plt.show()

# Rest of the code remains the same
# ...


# Plot
fig, axs = plt.subplots(ncols=6, nrows=7, figsize=(10, 12))
gs1 = axs[1, 0].get_gridspec()

# Remove the underlying axes
for ax in axs[1, :6]:
    ax.remove()

for ax in axs[2, :6]:
    ax.remove()

for ax in axs[3, :6]:
    ax.remove()

axbig1 = fig.add_subplot(gs1[1, :6])
axbig2=fig.add_subplot(gs1[2, :6])
axbig3=fig.add_subplot(gs1[3, :6])

# Plot 9 stimulus images side by side horizontally
for i, ax in enumerate(axs[0, :]):
    print(int(i * np.floor(900/5))-1)
    if i==0:
        ax.imshow(stimulus[5], cmap="gray", aspect="auto")
        ax.axis('off')
    else:
        ax.imshow(stimulus[int(i * np.floor(900/5))-1], cmap="gray", aspect="auto")
        ax.axis('off')


cmap="winter"
# Plot X_embedding in the bottom subplot
axbig1.imshow(emb_clip[:900], vmin=0, vmax=1.5, cmap=cmap, aspect="auto")
axbig2.imshow(emb_dino[:900], vmin=0, vmax=1.5, cmap=cmap, aspect="auto")
axbig3.imshow(emb_vit[:900], vmin=0, vmax=1.5, cmap=cmap, aspect="auto")
axbig1.set_yticks([])
axbig1.set_yticklabels([])
axbig2.set_yticks([])
axbig2.set_yticklabels([])
axbig3.set_yticks([])
axbig3.set_yticklabels([])
axbig1.axis('off')
axbig2.axis('off')
axbig3.axis('off')

# Draw vertical lines indicating the center of each image on X_embedding
for i in range(1, 6):
    print(i * np.floor(900/5))
    axbig1.axvline(x=i * np.floor(900/5)-1, color='red', linestyle='--', linewidth=3)
axbig1.axvline(x=5, color='red', linestyle='--', linewidth=3)

for i in range(1, 6):
    axbig2.axvline(x=i * np.floor(900/5)-1, color='orange', linestyle='--', linewidth=3)
axbig2.axvline(x=5, color='orange', linestyle='--', linewidth=3)
for i in range(1, 6):
    axbig3.axvline(x=i * np.floor(900/5)-1, color='yellow', linestyle='--',linewidth=3)
axbig3.axvline(x=5, color='yellow', linestyle='--',linewidth=3)


alpha=0.8
for i, ax in enumerate(axs[4, :]):
    print(int(i * np.floor(900/5))-1)
    if i==0:
        ax.plot(srtd_clip[5], color='red', alpha=alpha)
        ax.axis('off')
    else:
        ax.plot(srtd_clip[int(i * np.floor(900/5))-1], color='red',alpha=alpha)
        ax.axis('off')

# Plot 9 stimulus images side by side horizontally
for i, ax in enumerate(axs[5, :]):
    print(int(i * np.floor(900/5))-1)
    if i==0:
        ax.plot(srtd_dino[5], color='orange', alpha=alpha)
        ax.axis('off')
    else:
        ax.plot(srtd_dino[int(i * np.floor(900/5))-1], color='orange', alpha=alpha)
        ax.axis('off')

# Plot 9 stimulus images side by side horizontally
for i, ax in enumerate(axs[6, :]):
    print(int(i * np.floor(900/5))-1)
    if i==0:
        ax.plot(srtd_vit[5], color='yellow', alpha=alpha)
        ax.axis('off')
    else:
        ax.plot(srtd_vit[int(i * np.floor(900/5))-1], color='yellow', alpha=alpha)
        ax.axis('off')

fig.suptitle("Rastermap visualizations of CLIP (top), DINO (middle) and ViT (bottom) CLS tokens across movie frames, \n line plots are selected resorted raw CLS vectors (cross-sections)", fontsize=12)



fig.tight_layout()
plt.subplots_adjust(bottom=0)
plt.show()
