from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
from config import cache_path, save_path, embeddings_dct
import numpy as np 
import torch
from transformers import CLIPProcessor, CLIPModel, ViTImageProcessor, ViTModel
import os


class StimPrep:
    def __init__(self):
        self.movies_id='501704220' #This is three session A
        self.natural_scenes_id='501559087'
        self.boc = BrainObservatoryCache(manifest_file=str(
            cache_path / Path('brain_observatory_manifest.json')))
        '''
        [{'id': 501559087, 'imaging_depth': 175, 'targeted_structure': 'VISp', 'cre_line': 'Cux2-CreERT2', 
        'reporter_line': 'Ai93(TITL-GCaMP6f)', 'acquisition_age_days': 103, 'experiment_container_id': 511510736, 
        'session_type': 'three_session_B', 'donor_name': '222426', 'specimen_name': 'Cux2-CreERT2;Camk2a-tTA;Ai93-222426', 
        'fail_eye_tracking': True}, 
        
        {'id': 501704220, 'imaging_depth': 175, 'targeted_structure': 'VISp', 'cre_line': 'Cux2-CreERT2', 
        'reporter_line': 'Ai93(TITL-GCaMP6f)', 'acquisition_age_days': 104, 'experiment_container_id': 511510736, 
        'session_type': 'three_session_A', 'donor_name': '222426', 'specimen_name': 'Cux2-CreERT2;Camk2a-tTA;Ai93-222426', 
        'fail_eye_tracking': True}, 
        
        {'id': 501474098, 'imaging_depth': 175, 'targeted_structure': 'VISp', 'cre_line': 'Cux2-CreERT2', 
        'reporter_line': 'Ai93(TITL-GCaMP6f)', 'acquisition_age_days': 102, 'experiment_container_id': 511510736, 'session_type': 
        'three_session_C', 'donor_name': '222426', 'specimen_name': 'Cux2-CreERT2;Camk2a-tTA;Ai93-222426', 'fail_eye_tracking': True}]
        '''
        
    def load_data(self, raw_stim):
        stim_path = save_path/Path(raw_stim)
        stimulus=stim_path.split('.')[0]
        if os.path.exists(stim_path):
            stims=np.load(stim_path)
            return stims
        else:
            data_set = self.boc.get_ophys_experiment_data(self.movies_id)
            if stimulus=='natural_movie_one':
                movie_one = data_set.get_stimulus_template('natural_movie_one')
                np.save(stim_path, movie_one)
                stims=np.load(stim_path)
                return stims
        #data_set = self.boc.get_ophys_experiment_data(self.reference_eid)
        #self.movie_one = data_set.get_stimulus_template('natural_movie_one')
        #self.movie_two = data_set.get_stimulus_template('natural_movie_two')
        #self.movie_three = data_set.get_stimulus_template('natural_movie_three')
        #self.natural_scenes = data_set.get_stimulus_template('natural_scenes')

    def process_stims(self, stims, processor, model):
        n_stims = len(stims)
        embeddings = np.empty((n_stims, 768))
        for i in range(n_stims):
            print(i)
            inputs = processor(images=stims[i], return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            # print(outputs.keys())
            last_hidden_states = outputs.pooler_output.squeeze(0).detach().numpy()
            print(last_hidden_states.shape)
            embeddings[i, :] = last_hidden_states
        
    def make_dino(self):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
        model = ViTModel.from_pretrained('facebook/dino-vitb8')

    def make_clip(self,stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = CLIPProcessor.from_pretrained('facebook/dino-vitb8')
        model = CLIPModel.from_pretrained('facebook/dino-vitb8')
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings

    def make_embedding(self, emb_path, stim_path, model):
        #Make CLIP
        full_path = save_path / Path(emb_path)
        if os.path.exists(full_path):
            print(f'{full_path} already exists!')
        else:
            stims=self.load_data(stim_path)
            if model=='CLIP':
                embeddings = self.make_clip(stims)
                np.save(full_path, embeddings)