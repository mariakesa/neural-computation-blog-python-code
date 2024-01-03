from config import cache_path, save_path, embeddings_dct, stimuli_dct
from pathlib import Path
from make_embeddings import StimPrep
import os
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from config import cache_path, save_path
from pathlib import Path
from make_data import SingleEIDDat, Movie3
from regression import ridge_regression, make_visualizations
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
        for i, p in enumerate(embeddings_dct['vitmae']):
            emb_path = save_path / Path(p)
            raw_stim = embeddings_dct['raw_stims'][i]
            if os.path.exists(emb_path):
                print(f'ViTMAE {p} already exists!')
            else:
                self.stim_prep.make_embedding(emb_path, raw_stim, model='ViTMAE')
        for i, p in enumerate(embeddings_dct['vit']):
            emb_path = save_path / Path(p)
            raw_stim = embeddings_dct['raw_stims'][i]
            if os.path.exists(emb_path):
                print(f'ViT {p} already exists!')
            else:
                self.stim_prep.make_embedding(emb_path, raw_stim, model='ViT')

    def run_pipeline(self, eids):
        self.create_embeddings()
        self.eid_dat={}
        '''
        for eid in eids:
            dat=SingleEIDDat(eid)
            self.eid_dat[eid] = dat.make_train_test_data()
            print('Hello ',self.eid_dat[eid])
            cell_ids=dat.cell_ids
            df=pd.DataFrame()
            df['cell_ids']=cell_ids
            #print(self.eid_dat[eid])
            for trial in range(10):
                var_exp_clip=ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['clip']][trial])
                var_exp_dino =ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['dino']][trial])
                var_exp_vitmae =ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['vitmae']][trial])
                var_exp_vit =ridge_regression(self.eid_dat[eid][stimuli_dct['movie_one']['vit']][trial])
                df[f'var_exp_clip_{trial}']=var_exp_clip
                df[f'var_exp_dino_{trial}']=var_exp_dino
                df[f'var_exp_vitmae_{trial}']=var_exp_vitmae
                df[f'var_exp_vit_{trial}']=var_exp_vit
                if trial==9:
                    make_visualizations(cell_ids, self.eid_dat[eid][stimuli_dct['movie_one']['clip']][trial])
            df.to_csv('first_q_test.csv')
        #print(self.eid_dat)
        '''
        dat=Movie3(eids[0]).make_data_dct()
        
#Pipeline().run_pipeline([566752133])
#Session A
Pipeline().run_pipeline([501704220])

