from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from pathlib import Path
from config import cache_path, save_path, embeddings_dct
import numpy as np 
import torch
from transformers import CLIPVisionModel, ViTImageProcessor, ViTModel, AutoProcessor, AutoModel, AutoImageProcessor, ViTMAEModel, ResNetModel
import os


class StimPrep:
    def __init__(self):
        self.session_A=501704220 #This is three session A
        self.session_B=501559087
        self.session_C=501474098
        print(cache_path)
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(cache_path) / Path('brain_observatory_manifest.json')))
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
        stim_path = Path(save_path)/Path(raw_stim)
        stimulus=raw_stim.split('.')[0]
        if os.path.exists(stim_path):
            stims=np.load(stim_path)
            return stims
        else:
            if stimulus=='natural_movie_one':
                data_set = self.boc.get_ophys_experiment_data(self.session_A)
                movie_one = data_set.get_stimulus_template('natural_movie_one')
                np.save(stim_path, movie_one)
                stims=np.load(stim_path)
                return stims
            if stimulus=='natural_movie_two':
                data_set = self.boc.get_ophys_experiment_data(self.session_C)
                movie_two = data_set.get_stimulus_template('natural_movie_two')
                np.save(stim_path, movie_two)
                stims=np.load(stim_path)
                return stims
            if stimulus=='natural_movie_three':
                data_set = self.boc.get_ophys_experiment_data(self.session_A)
                movie_three= data_set.get_stimulus_template('natural_movie_three')
                np.save(stim_path, movie_three)
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
            #last_hidden_states = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
            #model_class_name = type(model).__name__
            #print(model_class_name)
            cls = outputs.pooler_output.squeeze().detach().numpy()
            #if model_class_name in ['CLIPVisionModel', 'ViTModel']:
                #cls = outputs.pooler_output.squeeze().detach().numpy()
            #elif model_class_name in ['ViTMAEModel']:
                #cls = np.mean(outputs.last_hidden_state.squeeze().detach().numpy(),axis=0)
            #print(cls.shape)
            #print(cls.shape)
            embeddings[i, :] = cls
        return embeddings
    
    def make_vit(self, stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch32-384')
        model = ViTModel.from_pretrained('google/vit-base-patch32-384')
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings
        
    def make_dino(self, stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb8')
        model = ViTModel.from_pretrained('facebook/dino-vitb8')
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings

    def make_clip(self, stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings
    
    def make_vitmae(self, stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings
    
    def make_resnet(self, stims):
        stims = np.repeat(stims[:, np.newaxis, :, :], 3, axis=1)
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetModel.from_pretrained("microsoft/resnet-50")
        embeddings  = self.process_stims(stims, processor, model)
        return embeddings

    def make_embedding(self, emb_path, stim_path, model):
        #Make CLIP
        full_path = Path(save_path) / Path(emb_path)
        if os.path.exists(full_path):
            print(f'{full_path} already exists!')
        else:
            stims=self.load_data(stim_path)
            if model=='CLIP':
                embeddings = self.make_clip(stims)
                np.save(full_path, embeddings)
            if model=='DINO':
                embeddings = self.make_dino(stims)
                np.save(full_path, embeddings)
            if model=='ViTMAE':
                pass
                #embeddings = self.make_vitmae(stims)
                #np.save(full_path, embeddings)
            if model=='ViT':
                embeddings=self.make_vit(stims)
                np.save(full_path, embeddings)