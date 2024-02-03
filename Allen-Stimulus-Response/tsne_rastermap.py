import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.graph_objects as go

# Load data
spks_clip = np.load("/media/maria/DATA/BrainObservatoryProcessedData/natural_movie_one_clip.npy").T.astype("float32")

X_embedded = TSNE(n_components=1, learning_rate='auto', init='random', perplexity=3).fit_transform(spks_clip).flatten()

print(X_embedded)
print(spks_clip.shape)
print(X_embedded.shape)

sortd=np.argsort(X_embedded).flatten()
print(sortd.shape)
print(sortd)

print(spks_clip[sortd,:])

z_data = spks_clip[sortd,:]

fig = go.Figure(data=[go.Surface(z=z_data, colorscale='sunset')])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()