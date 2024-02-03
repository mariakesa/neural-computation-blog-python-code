import numpy as np
emb_clip = np.load('emb_clip.npy')
print(emb_clip.shape)

import plotly.graph_objects as go

import pandas as pd

# Read data from a csv
z_data = emb_clip[:,:]

fig = go.Figure(data=[go.Surface(z=z_data, colorscale='sunset')])

fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()
