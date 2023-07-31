from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from transformers import ViTImageProcessor, ViTModel
import torch
from sklearn.decomposition import PCA

# Get data
# We want to get all the experiments with one cre line-- that's round enough for us
# We want to extract two train and validation sets-- select some number of unseen images
# We need extraction to give us data for movie one and natural images-- We are only targeting
# experiment B.
# Run regression experiments. Get data for all selected cell specimen and put them
# in the corresponding df.

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

    def regression(self, data_dct):
        np.random.seed = 7879
        train_images = np.random.randint(0, 118)
        train_movies = np.random.randint(0, 901)

    def average_responses_images(self, neural_responses, scene_stim_table):

        num_neurons, num_timepoints = neural_responses.shape
        num_frames = len(scene_stim_table)

        # Create an array to store the average response for each frame
        avg_responses = np.zeros((num_neurons, num_frames))

        # Group scene_stim_table by 'repeat' to handle frame repeats efficiently
        grouped_table = scene_stim_table.groupby('repeat')

        # Iterate over each group (i.e., frame repeat)
        for repeat, group in grouped_table:
            start_idx_list = group['start'].astype(int)
            end_idx_list = group['end'].astype(int)

            # Calculate the average response for the current frame repeat
            for i, (start_idx, end_idx) in enumerate(zip(start_idx_list, end_idx_list)):
                # The last frame overlaps with the beginning frame of the previous stimulus
                frame_response = neural_responses[:, start_idx:end_idx]
                avg_responses[:, group.index[i]] = frame_response.mean(axis=1)

        return avg_responses

    def average_responses_movie(self, neural_responses, movie_stim_table):

        num_neurons, _ = neural_responses.shape
        num_frames = 900

        # Create an array to store the average response for each frame
        avg_responses = np.zeros((num_neurons, num_frames))

        # Group scene_stim_table by 'repeat' to handle frame repeats efficiently
        grouped_table = movie_stim_table.groupby('frame')
        # print(grouped_table['start'])

        # Iterate over each group (i.e., frame repeat)
        for image_ind, group in grouped_table:
            ts = list(group['start'].astype(int))
            frame_response = neural_responses[:, ts]
            avg_responses[:, image_ind] = frame_response.mean(axis=1)
        return avg_responses

    def perform_pca(self, average_responses, num_components):
        # Initialize PCA with the desired number of components
        pca = PCA(n_components=num_components)

        # Fit the PCA model to the average_responses data and transform it
        reduced_responses = pca.fit_transform(average_responses.T)

        return reduced_responses.T

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
            avg_responses_movie = self.average_responses_movie(
                data_dct[eid]['neural_responses'], data_dct[eid]['movie_stim_table'])
            print(avg_responses_movie)
            # data_dct[eid]['natural_stim'] = self.data_set.get_stimulus_template(
            # 'natural_scenes')
            # print(np.unique(natural_stim_table.iloc[:, 0]))
        # self._make_regression_targets(data_dct)
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


MakeTestTrain('Emx1-IRES-Cre').fit_transform()
