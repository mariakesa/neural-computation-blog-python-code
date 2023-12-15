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
        for p in embeddings_dct['clip']:
            stim_path = save_path / Path(p)
            if os.path.exists(stim_path):
                print(f'CLIP {p} already exists!')
            else:
                stim_prep.make_embedding(stim_path, model='CLIP')
        

