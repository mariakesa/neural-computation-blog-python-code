import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import matplotlib.pyplot as plt
import os

output_dir = '/media/maria/DATA/AllenData'


def get_stims():
    output_dir = '/media/maria/DATA/AllenData'

    boc = BrainObservatoryCache(
        manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))

    # Choose random experiment-- the stimuli are the same
    Emx1_exps = boc.get_experiment_containers(cre_lines=['Emx1-IRES-Cre'])
    Emx1_id = Emx1_exps[0]['id']
    exps = boc.get_ophys_experiments(experiment_container_ids=[Emx1_id])
    exp = exps[0]
    data_set = boc.get_ophys_experiment_data(exp['id'])
    movie_stims = data_set.get_stimulus_template('natural_movie_one')
    scenes_stims = data_set.get_stimulus_template('natural_scenes')
    return movie_stims, scenes_stims


def do_PCA():
    # Paths
    absolute_path = os.path.dirname(__file__)
    relative_path1 = "dino_features/dino_movie_one.npy"
    relative_path2 = "dino_features/dino_natural_scenes.npy"
    full_path1 = os.path.join(absolute_path, relative_path1)
    full_path2 = os.path.join(absolute_path, relative_path2)

    dino_movie_feats = np.load(full_path1)
    dino_natural_scenes_feats = np.load(full_path2)

    stim_names = ['dino movies features', 'dino scene features']
    stims = [dino_movie_feats, dino_natural_scenes_feats]

    var_exp_lst = []
    for s in stims:
        pca = PCA(n_components=100).fit(s)
        var_exp = pca.explained_variance_ratio_
        var_exp_lst.append(var_exp)

    plt.plot(np.cumsum(var_exp_lst[0]), label=stim_names[0])
    plt.plot(np.cumsum(var_exp_lst[1]), label=stim_names[1])
    plt.legend(loc='lower right')
    plt.show()

    # print(dino_movie_feats.shape)


do_PCA()
