from pathlib import Path

cache_path = "/media/maria/DATA/AllenData"
save_path = "/media/maria/DATA/BrainObservatoryProcessedData"

embeddings_dct = {
    'raw_stims': ['natural_movie_one.npy'],#, 'natural_movie_two.npy', 'natural_movie_three.npy', 'natural_natural_scenes.npy'],
    'clip': ['natural_movie_one_clip.npy'],#, 'natural_movie_two_clip.npy', 'natural_movie_three_clip.npy', 'natural_natural_scenes_clip.npy']
    'dino': ['natural_movie_one_dino.npy']
}

stimuli_dct = {
    'movie_one': {'clip': 'natural_movie_one_clip.npy',
         'dino': 'natural_movie_one_dino.npy'}
}