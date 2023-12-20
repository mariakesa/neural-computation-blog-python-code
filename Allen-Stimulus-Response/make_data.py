from config import cache_path, save_path, embeddings_dct, stimuli_dct
from pathlib import Path
from make_embeddings import StimPrep
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

class SingleEIDDat:
    def __init__(self, eid):
        boc = BrainObservatoryCache(manifest_file=str(
                Path(cache_path) / 'brain_observatory_manifest.json'))
        
        self.dataset = boc.get_ophys_experiment_data(eid)

    def make_data_dct(self):
        self.data_dct={}
        self.data_dct['movie_stim_table'] = self.dataset.get_stimulus_table(
                    'natural_movie_one')

        self.data_dct['neural_responses'] = self.dataset.get_dff_traces()[1]

    def get_embeddings(self):
        self.embeddings={}
        for stimulus in stimuli_dct.keys():
            print(stimuli_dct[stimulus].keys())
            for model in stimuli_dct[stimulus].keys():
                self.embeddings[stimuli_dct[stimulus][model]] = np.load(Path(save_path)/Path(stimuli_dct[stimulus][model]))

    def make_regression_data(self, embedding):
        np.random.seed = 7879
        stimuli = self.data_dct['movie_stim_table'].loc[self.data_dct['movie_stim_table']['repeat'] == 9]
        print(stimuli)

        X_train, X_test, y_train_inds, y_test_inds = train_test_split(embedding,stimuli['start'].values, test_size=0.7, random_state=42)
        y_train=self.data_dct['neural_responses'][:,y_train_inds]
        y_test=self.data_dct['neural_responses'][:,y_test_inds]

        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
        
    def make_train_test_data(self):
        self.make_data_dct()
        self.get_embeddings()
        train_test_data={}
        for model in self.embeddings.keys():
            embedding = self.embeddings[model]
            train_test_data[model] = self.make_regression_data(embedding)
        print('Boom!', train_test_data)
        return train_test_data