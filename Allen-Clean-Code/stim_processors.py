from config import cache_path, save_path, embeddings_dct, stimuli_dct, stimulus_session_dict
from pathlib import Path
from make_embeddings import StimPrep
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

#rng = np.random.default_rng(77)

class ProcessMovieRecordings:
    def __init__(self):
        self.boc = BrainObservatoryCache(manifest_file=str(
                Path(cache_path) / 'brain_observatory_manifest.json'))
        
        self.eid_dict = self.make_container_dict()
        #self.dataset = boc.get_ophys_experiment_data(eid)
        #self.cell_ids = self.dataset.get_cell_specimen_ids()
        #self.stimulus = stimulus

    def make_container_dict(self):
        '''
        Parses which experimental id's (values)
        correspond to which experiment containers (keys).
        '''
        experiment_container = self.boc.get_experiment_containers()
        container_ids=[dct['id'] for dct in experiment_container]
        eids=self.boc.get_ophys_experiments(experiment_container_ids=container_ids)
        df=pd.DataFrame(eids)
        reduced_df=df[['id', 'experiment_container_id', 'session_type']]
        grouped_df = df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
        eid_dict = {}
        for row in grouped_df.itertuples(index=False):
            container_id, session_type, ids = row
            if container_id not in eid_dict:
                eid_dict[container_id] = {}
            eid_dict[container_id][session_type] = ids[0]
        return eid_dict


    def make_data_dct(self, data_dct, dataset, stimulus):
        data_dct['movie_stim_table_'+stimulus] = dataset.get_stimulus_table(stimulus)
        return data_dct


    def get_embeddings(self):
        self.embeddings={}
        for model in stimuli_dct[self.stimulus]:
            self.embeddings[stimuli_dct[self.stimulus][model]] = np.load(Path(save_path)/Path(stimuli_dct[self.stimulus][model]))
    

    def get_session_data(self, dataset, session):
        
        data_dct['neural_responses'] = dataset.get_dff_traces()[1]
        #stimuli = stimulus_session_dict[session]
        #for stim in stimuli:
        


    def make_all_data(self,container_id):
        self.get_embeddings()
        data_dct = {}
        current_container_dict=self.eid_dict[container_id]
        for session in self.current_container_dict.keys():
            data_dct[container_id][session] = self.make_regression_data(container_id, session)

    def process_single_trial(self, movie_stim_table, dff_traces, trial, embedding):
        stimuli = movie_stim_table.loc[self.data_dct['movie_stim_table']['repeat'] == trial]
        X_train, X_test, y_train_inds, y_test_inds = train_test_split(embedding,stimuli['start'].values, test_size=0.7, random_state=42)
        y_train= dff_traces[:,y_train_inds]
        y_test= dff_traces['neural_responses'][:,y_test_inds]
        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
        


    def make_regression_data(self, container_id, session):
        session_eid  = self.current_container_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = stimulus_session_dict[session]
        session_dct = {}
        for s in session_stimuli:
            movie_stim_table = dataset.get_stimulus_table(s)
            for trial in range(10):
                session_dct[str(s)+'_'+str(trial)] = self.process_single_trial(movie_stim_table, dff_traces, trial)
        return session_dct

        trials_dct={}
        for trial in range(10):
            np.random.seed = 7879
            stimuli = self.data_dct['movie_stim_table'].loc[self.data_dct['movie_stim_table']['repeat'] == trial]
            if trial==0:
                print(stimuli)

            X_train, X_test, y_train_inds, y_test_inds = train_test_split(embedding,stimuli['start'].values, test_size=0.7, random_state=42)
            y_train=self.data_dct['neural_responses'][:,y_train_inds]
            y_test=self.data_dct['neural_responses'][:,y_test_inds]

            trials_dct[trial]={'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
            



        #train_test_data={}
        #for model in self.embeddings.keys():
            #embedding = self.embeddings[model]
            #train_test_data[model] = self.make_regression_data(embedding)
        #print('Boom!', train_test_data)
        #return train_test_data

ProcessMovieRecordings().make_train_test_data(511510736)

class SingleEIDDat:
    def __init__(self, eid):
        boc = BrainObservatoryCache(manifest_file=str(
                Path(cache_path) / 'brain_observatory_manifest.json'))
        
        self.dataset = boc.get_ophys_experiment_data(eid)
        self.cell_ids = self.dataset.get_cell_specimen_ids()
        print('My cell id\'s: ', self.cell_ids)
        #print(self.dataset.)

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
        trials_dct={}
        for trial in range(10):
            np.random.seed = 7879
            stimuli = self.data_dct['movie_stim_table'].loc[self.data_dct['movie_stim_table']['repeat'] == trial]
            if trial==0:
                print(stimuli)

            X_train, X_test, y_train_inds, y_test_inds = train_test_split(embedding,stimuli['start'].values, test_size=0.7, random_state=42)
            y_train=self.data_dct['neural_responses'][:,y_train_inds]
            y_test=self.data_dct['neural_responses'][:,y_test_inds]

            trials_dct[trial]={'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
        return trials_dct
        
    def make_train_test_data(self):
        self.make_data_dct()
        self.get_embeddings()
        train_test_data={}
        for model in self.embeddings.keys():
            embedding = self.embeddings[model]
            train_test_data[model] = self.make_regression_data(embedding)
        #print('Boom!', train_test_data)
        return train_test_data
    
