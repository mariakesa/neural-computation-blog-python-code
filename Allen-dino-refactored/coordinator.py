#Embed stimuli--> stimuli data structure
#
#https://github.com/AllenInstitute/brain_observatory_examples/blob/master/Visual%20Coding%202P%20Cheat%20Sheet%20October2018.pdf
import json
from abc import ABC, abstractmethod
from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np
from transformers import ViTImageProcessor, ViTModel
import torch

class Stimulus:
    def __init__(self):
        self.cache_path=self.read_cache_path()
        print(self.cache_path)
        self.save_path=self.read_save_path()

    def read_cache_path(self):
        f = open('config.json')
        config = json.load(f)
        f.close()
        return config['cache_path']

    def read_save_path(self):
        f = open('config.json')
        config = json.load(f)
        f.close()
        return config['save_path']

    def write_paths_to_json(self):
        f = open('paths.json', "r")
        path_dct = json.load(f)
        path_dct['raw_stims']= [{"path": str(Path(self.save_path) / 'movie_one.npy'), "stim_name": "movie_one"}]
        f.close()
        f = open('paths.json', "w")
        json.dump(path_dct, f, indent=4)
        f.close()

    def save_stims(self):
        boc = BrainObservatoryCache(
            manifest_file=str(Path(self.cache_path) / 'brain_observatory_manifest.json'))

        
        #session_A = boc.get_experiment_containers(session_types=['three_session_A'])
        #id=session_A[0]['id']
        exps = boc.get_ophys_experiments(session_types=['three_session_A'])
        # pick one of the cux2 experiment containers
        #cux2_ec_id = cux2_ecs[-1]['id']

        # Find the experiment with the static static gratings stimulus
        #exps = boc.get_ophys_experiments(experiment_container_ids=[cux2_ec_id], 
                                        #stimuli=[stim_in])[0]
        exp = exps[0]
        #print(exp)
        data_set = boc.get_ophys_experiment_data(exp['id'])
        print(data_set)
        movie_one = np.array(data_set.get_stimulus_template('natural_movie_one'))
        print(movie_one.shape)
        #movie_two = data_set.get_stimulus_template('natural_movie_two')
        movie_three = data_set.get_stimulus_template('natural_movie_three')
        np.save(str(Path(self.save_path)/ 'movie_one.npy'), movie_one)

        self.write_paths_to_json()
        '''
        #print(Emx1_exps)
        Emx1_id = Emx1_exps[0]['id']
        exps = boc.get_ophys_experiments(experiment_container_ids=[Emx1_id])
        exp = exps[0]
        #print(exp)
        data_set = boc.get_ophys_experiment_data(exp['id'])
        #print(data_set)
        movie_one = data_set.get_stimulus_template('natural_movie_one')
        #movie_two = data_set.get_stimulus_template('natural_movie_two')
        movie_three = data_set.get_stimulus_template('natural_movie_three')
        scenes_stims = data_set.get_stimulus_template('natural_scenes')
        '''
        #return movie_stims, scenes_stims

#a=Stimulus().save_stims()
class StimulusEmbedding(ABC):
    pass
    #DINO
    #VisionTransformer
    #PCA

    #There are three natural movies
    #Embed the images and save them.
    #The output is going to be some array.
    pass

    def read_cache_path(self):
        f = open('config.json')
        config = json.load(f)
        f.close()
        return config['cache_path']

    def read_save_path(self):
        f = open('config.json')
        config = json.load(f)
        f.close()
        return config['save_path']

    def read_paths(self):
        f = open('paths.json')
        paths = json.load(f)
        f.close()
        stim_paths=paths['raw_stims']
        return stim_paths

    @abstractmethod
    def get_embedding(self):
        pass

    @abstractmethod
    def write_paths_to_json(self):
        pass




class DINOEmbedding(StimulusEmbedding):

    def __init__(self):
        self.cache_path=self.read_cache_path()
        print(self.cache_path)
        self.save_path=self.read_save_path()
        self.stims_path=self.read_paths()

    def write_paths_to_json(self):
        f = open('paths.json', "r")
        path_dct = json.load(f)
        path_dct['dino_fatures']={"path":str(Path(self.save_path)/"dino_movie_one.npy"), 
                                    "features_name":"dino_movie_one" }
        print(path_dct)
        f.close()
        f = open('paths.json', "w")
        json.dump(path_dct, f)
        f.close()

    def get_embedding(self, stims):
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

    def all_embeddings(self):
        for stim in self.stims_path:
            stims=np.load(stim["path"])
            dino_features=self.get_embedding(stims)
            path_str='dino_'+stim["stim_name"]
            np.save(str(Path(self.save_path)/path_str))
            print(dino_features)
        self.write_paths_to_json()

    
        

a=DINOEmbedding().all_embeddings()









class StructureData:
    #Multi-file embedding
    #Single-file embedding

    #Train and Test

    #Raw data, PC's
    pass

class RegressionModel:
    #OLS
    #Ridge--> optimizable hyperparameters, autosklearn?
    #NN--> architecture search
    #PartialLeastSquares
    pass

class Metric:
    #Variance explained
    #R2
    pass


class Coordinator:
    #DataPreprocessing step-- cache?
    #Regression model
    pass

print('y')

