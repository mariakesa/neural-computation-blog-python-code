from config import cache_path, save_path, embeddings_dct
from pathlib import Path
from make_embeddings import StimPrep
import os


class Pipeline:
    def __init__(self):
        self.stim_prep=StimPrep()

    def create_embeddings(self):
        pass

    def run_pipeline(self):
        #Check if stimulus embeddings exist, start with CLIP
        for i, p in enumerate(embeddings_dct['clip']):
            emb_path = save_path / Path(p)
            raw_stim = embeddings_dct['raw_stims'][i]
            if os.path.exists(emb_path):
                print(f'CLIP {p} already exists!')
            else:
                self.stim_prep.make_embedding(emb_path, raw_stim, model='CLIP')
        
Pipeline().run_pipeline()

