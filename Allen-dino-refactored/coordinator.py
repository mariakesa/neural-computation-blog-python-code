#Embed stimuli--> stimuli data structure
#
import json
from abc import ABC, abstractmethod

class StimulusEmbedding(ABC):
    pass
    #DINO
    #VisionTransformer
    #PCA

    #Embed the images and save them.
    #The output is going to be some array.
    pass

    def read_cache_path(self):
        f = open('config.json')
        config = json.load(f)
        f.close()
        return config['cache_path']

    @abstractmethod
    def write_path_to_json(self):
        pass


class DINOEmbedding(StimulusEmbedding):

    def __init__(self):
        self.cache_path=self.read_cache_path()
        print(self.cache_path)


    def write_path_to_json(self):
        f = open('paths.json', "r")
        path_dct = json.load(f)
        path_dct['dino']='something-something'
        print(path_dct)
        f.close()
        f = open('paths.json', "w")
        json.dump(path_dct, f)
        f.close()
        

a=DINOEmbedding().write_path_to_json()









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

