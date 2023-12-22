from config import cache_path, save_path, embeddings_dct, stimuli_dct
from pathlib import Path
from make_embeddings import StimPrep
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from config import cache_path, save_path
from pathlib import Path
from make_data import SingleEIDDat
from regression import ridge_regression
import pandas as pd

class Pipeline:
    def __init__(self):
        self.stim_prep=StimPrep()
        

    def create_embeddings(self):
        #Check if stimulus embeddings exist, start with CLIP
        for i, p in enumerate(embeddings_dct['clip']):
            emb_path = save_path / Path(p)
            raw_stim = embeddings_dct['raw_stims'][i]
            if os.path.exists(emb_path):
                print(f'CLIP {p} already exists!')
            else:
                self.stim_prep.make_embedding(emb_path, raw_stim, model='CLIP')
        for i, p in enumerate(embeddings_dct['dino']):
            emb_path = save_path / Path(p)
            raw_stim = embeddings_dct['raw_stims'][i]
            if os.path.exists(emb_path):
                print(f'DINO {p} already exists!')
            else:
                self.stim_prep.make_embedding(emb_path, raw_stim, model='DINO')

    def run_pipeline(self, eids):
        self.create_embeddings()
        self.eid_dat={}
        for eid in eids:
            dat=SingleEIDDat(eid)
            self.eid_dat[eid] = dat.make_train_test_data()
            cell_ids=dat.cell_ids
            print(self.eid_dat[eid])
            var_exp_clip=ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['clip']])
            var_exp_dino =ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['dino']])
            df=pd.DataFrame()
            df['cell_ids']=cell_ids
            df['var_exp_clip']=var_exp_clip
            df['var_exp_dino']=var_exp_dino
            df.to_csv('first_q_test.csv')
        #print(self.eid_dat)

        
Pipeline().run_pipeline([566752133])

