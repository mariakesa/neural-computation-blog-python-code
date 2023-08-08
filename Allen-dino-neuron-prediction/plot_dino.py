import numpy as np
import matplotlib.pyplot as plt
'''
dino_features_natural_movie = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_movie_one.npy')
dino_features_dino_natural_scenes = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_natural_scenes.npy')
# print(features.shape)

# Create subplots with two heatmaps side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Heatmap for dino_features_natural_movie
axs[0].imshow(dino_features_natural_movie.T, cmap='tab20b', aspect='auto')
axs[0].set_title('DINO Features - Natural Movie')
axs[0].set_ylabel('Feature Dimension')
axs[0].set_xlabel('Sample Index')
# axs[0].set_xticks([])  # Remove x-axis ticks for cleaner visualization
# axs[0].set_yticks([])  # Remove y-axis ticks for cleaner visualization

# Heatmap for dino_features_dino_natural_scenes
axs[1].imshow(dino_features_dino_natural_scenes.T,
              cmap='tab20b', aspect='auto')
axs[1].set_title('DINO Features - Dino Natural Scenes')
axs[1].set_ylabel('Feature Dimension')
axs[1].set_xlabel('Sample Index')
# axs[1].set_xticks([])  # Remove x-axis ticks for cleaner visualization
# axs[1].set_yticks([])  # Remove y-axis ticks for cleaner visualization

# Adjust spacing between subplots
plt.tight_layout()

plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Load DINO features for two types of stimulus sequences
dino_features_natural_movie = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_movie_one.npy')
dino_features_dino_natural_scenes = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_natural_scenes.npy')

# Combine the features from both sequences
combined_features = np.concatenate(
    (dino_features_natural_movie, dino_features_dino_natural_scenes), axis=0)

# Apply PCA for dimensionality reduction to 3 components
pca = PCA(n_components=3)
reduced_features = pca.fit_transform(combined_features)

# Separate the reduced features back into two sets
reduced_features_natural_movie = reduced_features[:
                                                  dino_features_natural_movie.shape[0]]
reduced_features_dino_natural_scenes = reduced_features[dino_features_natural_movie.shape[0]:]

# Create a 3D PCA plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot features for natural movie sequence
ax.scatter(
    reduced_features_natural_movie[:, 0],
    reduced_features_natural_movie[:, 1],
    reduced_features_natural_movie[:, 2],
    c='r',
    label='Natural Movie'
)

# Plot features for dino natural scenes sequence
ax.scatter(
    reduced_features_dino_natural_scenes[:, 0],
    reduced_features_dino_natural_scenes[:, 1],
    reduced_features_dino_natural_scenes[:, 2],
    c='b',
    label='Dino Natural Scenes'
)

ax.set_title('3D PCA Plot of DINO Features')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()

plt.show()
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load DINO features for two types of stimulus sequences
dino_features_natural_movie = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_movie_one.npy')
dino_features_dino_natural_scenes = np.load(
    '/home/maria/neural-computation-blog-python-code/Allen-dino-neuron-prediction/dino_features/dino_natural_scenes.npy')

# Apply separate PCA for each dataset
pca_natural_movie = PCA(n_components=3)
pca_dino_natural_scenes = PCA(n_components=3)

reduced_features_natural_movie = pca_natural_movie.fit_transform(
    dino_features_natural_movie)
reduced_features_dino_natural_scenes = pca_dino_natural_scenes.fit_transform(
    dino_features_dino_natural_scenes)

# Create a 3D PCA plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot features for natural movie sequence
ax.scatter(
    reduced_features_natural_movie[:, 0],
    reduced_features_natural_movie[:, 1],
    reduced_features_natural_movie[:, 2],
    c='r',
    label='Natural Movie'
)

# Plot features for dino natural scenes sequence
ax.scatter(
    reduced_features_dino_natural_scenes[:, 0],
    reduced_features_dino_natural_scenes[:, 1],
    reduced_features_dino_natural_scenes[:, 2],
    c='b',
    label='Dino Natural Scenes'
)

ax.set_title('3D PCA Plot of DINO Features')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.legend()

plt.show()
