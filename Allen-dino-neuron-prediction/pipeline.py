from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from transformers import ViTImageProcessor, ViTModel
import torch

# Get data
# We want to get all the experiments with one cre line-- that's round enough for us
# We want to extract two train and validation sets-- select some number of unseen images
# We need extraction to give us data for movie one and natural images-- We are only targeting
# experiment B.
# Run regression experiments. Get data for all selected cell specimen and put them
# in the corresponding df.
# Regression experiments -- record variance explained on test set:
# Per repeat, for mean
# For raw images and videos
# For DINO features images and videos
# For PCA features of images and videos

output_dir = '/media/maria/DATA/AllenData'

# Get data


class MakeTestTrain():
    def __init__(self, cre_line):
        self.cre_line = cre_line
        self.output_dir = '/media/maria/DATA/AllenData'
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(output_dir) / 'brain_observatory_manifest.json'))

    def _get_eids(self):
        self.experiment_container = self.boc.get_experiment_containers(cre_lines=[
                                                                       self.cre_line])
        eids = []
        z = 0
        for e in self.experiment_container:
            if z < 1:
                id = e['id']
                exps = self.boc.get_ophys_experiments(
                    experiment_container_ids=[id])
                for i in exps:
                    if i['session_type'] == 'three_session_B':
                        eids.append(i['id'])
            z += 1
        print(eids)
        return eids

    def _make_regression_targets(self, data_dct):
        np.random.seed = 7879
        train_images = np.random.randint(0, 118)
        train_movies = np.random.randint(0, 901)
        for k in data_dct.keys():

            stimuli = data_dct[k]['movie_stim_table'].loc[data_dct[k]
                                                          ['movie_stim_table']['repeat'] == 2]
            test_movies = np.random.choice(range(900), 200, replace=False)
            # Generate an array containing all possible indices from 0 to 900 (inclusive)
            all_indices = np.arange(900)
            # Get the test indices as the complement of the train indices
            train_movies = np.setdiff1d(all_indices, test_movies)
            train_inds = stimuli.loc[stimuli['frame'].isin(
                train_movies), 'start'].values
            test_inds = stimuli.loc[stimuli['frame'].isin(
                test_movies), 'start'].values

            y_train_movie = data_dct[k]['neural_responses'][:, train_inds]
            y_test_movie = data_dct[k]['neural_responses'][:, test_inds]

    def fit_transform(self):
        eids = self._get_eids()
        data_dct = {}
        for eid in eids:
            data_dct[eid] = {}
        for eid in eids:
            data_set = self.boc.get_ophys_experiment_data(eid)
            data_dct[eid]['movie_stim_table'] = data_set.get_stimulus_table(
                'natural_movie_one')
            # data_dct[eid]['movie_stim'] = self.data_set.get_stimulus_template(
            # 'natural_movie_one')
            # Make an extra column with the number of the repeat in the scenes
            # table
            df_scene_table = data_set.get_stimulus_table(
                'natural_scenes')
            df_sorted = df_scene_table.sort_values(
                by=["frame", "start"], ascending=[True, True])
            df_sorted['repeat'] = df_sorted.groupby('frame').cumcount()
            data_dct[eid]['scene_stim_table'] = df_sorted
            data_dct[eid]['neural_responses'] = data_set.get_dff_traces()[1]
            # data_dct[eid]['natural_stim'] = self.data_set.get_stimulus_template(
            # 'natural_scenes')
            # print(np.unique(natural_stim_table.iloc[:, 0]))
        self._make_regression_targets(data_dct)
        '''
        # Check whether stimulus templates are the same
        for k in data_dct.keys():
            for l in data_dct.keys():
                if k != l:
                    if np.array_equal(data_dct[k]['movie_stim'], data_dct[l]['movie_stim']):
                        print(True)
                    else:
                        print(False)
                    if np.array_equal(data_dct[k]['natural_stim'], data_dct[l]['natural_stim']):
                        print(True)
                    else:
                        print(False)
        '''


def get_dino_features_for_stims(stims):
    '''
    Extract dino features from stims.
    '''
    stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
    processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
    model = ViTModel.from_pretrained('facebook/dino-vitb8')

    n_stims = len(stims)
    dino_features = np.empty((n_stims, 768))
    for i in range(n_stims):
        print(i)
        inputs = processor(images=stims[i], return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # print(outputs.keys())
        last_hidden_states = outputs.pooler_output.squeeze(0).detach().numpy()
        print(last_hidden_states.shape)
        dino_features[i, :] = last_hidden_states

    return dino_features


MakeTestTrain('Emx1-IRES-Cre').fit_transform()
