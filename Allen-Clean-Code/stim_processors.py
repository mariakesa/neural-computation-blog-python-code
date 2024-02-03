from config import cache_path, save_path, embeddings_dct, stimuli_dct, stimulus_session_dict
from pathlib import Path
#from make_embeddings import StimPrep
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import traceback

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.linalg import null_space
import numpy as np
import json, codecs


def ridge_regression(dat_dct):

    y_train, y_test, X_train, X_test= dat_dct['y_train'], dat_dct['y_test'], dat_dct['X_train'], dat_dct['X_test']

    regr=Ridge(10)

    # Fit the model with scaled training features and target variable
    regr.fit(X_train, y_train.T)

    # Make predictions on scaled test features
    predictions = regr.predict(X_test)

    scores=[]
    for i in range(0,y_test.shape[0]):
        scores.append(r2_score(y_test.T[:,i], predictions[:,i]))
    return scores, regr.coef_.tolist()

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
        cell_ids = dataset.get_cell_specimen_ids()
        dff_traces = dataset.get_dff_traces()[1]
        session_stimuli = stimulus_session_dict[session]
        session_dct = pd.DataFrame()
        regression_vec_dct={}
        session_dct['cell_ids'] = cell_ids
        #regression_vec_dct['cell_ids'] = cell_ids
        #Compile the sessions into the same column to avoind NAN's
        #and make the data processing a bit easier
        if session=='three_session_C2':
            sess='three_session_C'
        else:
            sess=session
        for s in session_stimuli:
            for m in stimuli_dct[s].keys():
                movie_stim_table = dataset.get_stimulus_table(s)
                embedding=self.embeddings[stimuli_dct[s][m]]
                for trial in range(10):
                    random_state=self.random_state_dct[session][s][trial]
                    data=self.process_single_trial(movie_stim_table, dff_traces, trial, embedding, random_state=random_state)
                    #Code: session-->model-->stimulus-->trial
                    var_exps,regr_vecs=ridge_regression(data)
                    session_dct[str(sess)+'_'+str(m)+'_'+str(s)+'_'+str(trial)] = var_exps
                    regression_vec_dct[str(sess)+'_'+str(m)+'_'+str(s)+'_'+str(trial)]=regr_vecs
        return session_dct, regression_vec_dct
    
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
                processor.make_regression_data(container_id, s)
            except Exception as e:
                print(f"Error processing container {container_id}, session {s}: {e}")
                continue
        cnt+=1

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
                processor.make_regression_data(container_id, s)
            except Exception as e:
                print(f"Error processing container {container_id}, session {s}: {e}")
                continue
        cnt+=1

def make_df():
    def compile_dfs(sess_dct):
        # Initialize an empty DataFrame to store the merged result
        merged_df = pd.DataFrame()

        # Iterate through each key in sess_dct
        for k in sess_dct.keys():
            # Retrieve the DataFrame associated with the key
            df = sess_dct[k]

            # Check if merged_df is empty (first iteration)
            if merged_df.empty:
                merged_df = df
            else:
                # Merge the current DataFrame with the existing merged_df based on 'cell_ids' column
                merged_df = pd.merge(merged_df, df, on='cell_ids', how='inner')

        return merged_df
            
    output_dir = '/media/maria/DATA/AllenData'
    boc = BrainObservatoryCache(manifest_file=str(Path(output_dir) / 'brain_observatory_manifest.json'))
    experiment_container = boc.get_experiment_containers()
    rng = np.random.default_rng(78)
    exp_ids=[dct['id'] for dct in experiment_container]
    random_exp_ids = rng.choice(exp_ids, size=100, replace=False)
    random_exp_ids = [511510736]
    sessions=['three_session_A', 'three_session_B', 'three_session_C', 'three_session_C2']
    processor=ProcessMovieRecordings()
    sess_dct={}
    cnt=0
    regr_vec_dct={}
    for container_id in random_exp_ids:
        print(cnt)
        for s in sessions:
            try:
                df, regr_vec_df=processor.make_regression_data(container_id, s)
                sess_dct[s]=df
                regr_vec_dct[s]=regr_vec_df
            except Exception as e:
                print(f"Error processing container {container_id}, session {s}: {e}")
                #traceback.print_exc()
                continue
        cnt+=1
    results=compile_dfs(sess_dct)
    #regr_dims=compile_dfs(regr_vec_dct)
    results.to_csv('test.csv')
    #regr_vec_dct.to_json('regr_dims.json')
    #https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    #json.dump(regr_vec_dct, codecs.open("regr_dims_test.json", 'w', encoding='utf-8'), 
          #separators=(',', ':'), 
          #sort_keys=True, 
          #indent=4)
    


#start=time.time()
#pull_data()
#end=time.time()
#print('100 pulls time: ', end-start)

start=time.time()
make_df()
end=time.time()
print('100 pulls time: ', end-start)