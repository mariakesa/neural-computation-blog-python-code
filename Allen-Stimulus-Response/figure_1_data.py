import numpy as np
from config import save_path,stimuli_dct
from pathlib import Path
import matplotlib.pyplot as plt
import json

def make_embedding_json(movie):
    paths=stimuli_dct[movie]
    path=Path(save_path)/Path(paths['clip'])
    my_dat=np.load(path)
    #for i in range(0,my_dat.shape[0]-1):
        #plt.plot(my_dat[i,:]-my_dat[i+1])
        #plt.show()
    previous_step_diff = []
    #for i in range(1, my_dat.shape[0]):
        #plt.scatter(my_dat[i-1,:],my_dat[i])
        #plt.show()
    json_dct={}
    for i in range(1,my_dat.shape[0]):
        json_dct[i]=list(my_dat[i-1]-my_dat[i])
        plt.plot(json_dct[i])
        plt.show()
        print(len(json_dct[i]))
    #json_path=Path(save_path)/Path("clip_emb_test.json")
    #with open(json_path, "w") as outfile: 
        #json.dump(json_dct, outfile, indent=4)


make_embedding_json('movie_one')



