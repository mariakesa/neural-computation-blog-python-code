import numpy as np
import matplotlib.pyplot as plt

dat=np.load('emb_clip.npy')

plt.imshow(dat,cmap="winter",aspect="auto")
plt.show()