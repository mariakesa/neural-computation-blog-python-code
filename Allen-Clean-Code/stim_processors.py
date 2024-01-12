from config import cache_path, save_path, embeddings_dct, stimuli_dct, stimulus_session_dict
from pathlib import Path
#from make_embeddings import StimPrep
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
        self.random_state_dct=self.generate_random_state()
        self.embeddings = self.get_embeddings()

    def generate_random_state(self):
        np.random.seed(7)

        # Function to generate a random integer
        def generate_random_integer():
            return np.random.randint(1, 101)  # Generates a random integer between 1 and 100 (inclusive)

        # Given stimulus_session_dict
        stimulus_session_dict = {
            'three_session_A': ['natural_movie_one', 'natural_movie_three'],
            'three_session_B': ['natural_movie_one'],
            'three_session_C': ['natural_movie_one', 'natural_movie_three'],
            'three_session_C2': ['natural_movie_one', 'natural_movie_three']
        }

        # Create the main dictionary
        random_state_dct = {}

        # Populate the dictionary using stimulus_session_dict
        for session, stimuli_list in stimulus_session_dict.items():
            session_dict = {}
            for stimulus in stimuli_list:
                nested_dict = {trial: generate_random_integer() for trial in range(10)}
                session_dict[stimulus] = nested_dict
            random_state_dct[session] = session_dict
        return random_state_dct

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
        embeddings={}
        for s in stimuli_dct.keys():
            for model in stimuli_dct[s]:
                embeddings[stimuli_dct[s][model]] = np.load(Path(save_path)/Path(stimuli_dct[s][model]))  
        return embeddings    

    def process_single_trial(self, movie_stim_table, dff_traces, trial, embedding, random_state):
        stimuli = movie_stim_table.loc[movie_stim_table['repeat'] == trial]
        X_train, X_test, y_train_inds, y_test_inds = train_test_split(embedding,stimuli['start'].values, test_size=0.7, random_state=random_state)
        y_train= dff_traces[:,y_train_inds]
        y_test= dff_traces[:,y_test_inds]
        return {'y_train': y_train, 'y_test': y_test, 'X_train': X_train, 'X_test': X_test}
        


    def make_regression_data(self, container_id, session):
        session_eid  = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = stimulus_session_dict[session]
        session_dct = {}
        for s in session_stimuli:
            for m in stimuli_dct[s].keys():
                movie_stim_table = dataset.get_stimulus_table(s)
                embedding=self.embeddings[stimuli_dct[s][m]]
                for trial in range(10):
                    random_state=self.random_state_dct[session][s][trial]
                    session_dct[str(m)+'_'+str(s)+'_'+str(trial)] = self.process_single_trial(movie_stim_table, dff_traces, trial, embedding, random_state=random_state)
        return session_dct
    
import time
start=time.time()
a=ProcessMovieRecordings().make_regression_data(511510736, 'three_session_A')
end=time.time()
print(end-start)
#print(a.keys())

def pull_data():
    output_dir = '/media/maria/DATA/AllenData'
    boc = BrainObservatoryCache(manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    experiment_container = boc.get_experiment_containers()
    rng = np.random.default_rng(78)
    exp_ids=[dct['id'] for dct in experiment_container]
    random_exp_ids = rng.choice(exp_ids, size=100, replace=False)
    sessions=['three_session_A', 'three_session_B', 'three_session_C', 'three_session_C2']
    processor=ProcessMovieRecordings()
    cnt=0
    for container_id in random_exp_ids:
        print(cnt)
        for s in sessions:
            try:
                if s == 'three_session_C' or s == 'three_session_C2':
                    processor.make_regression_data(container_id, s)
                else:
                    processor.make_regression_data(container_id, s)
            except Exception as e:
                print(f"Error processing container {container_id}, session {s}: {e}")
                continue
        cnt+=1


start=time.time()
pull_data()
end=time.time()
print('100 pulls time: ', end-start)
