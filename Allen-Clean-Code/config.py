from pathlib import Path

cache_path = "/media/maria/DATA/AllenData"
save_path = "/media/maria/DATA/BrainObservatoryProcessedData"

embeddings_dct = {
    'raw_stims': ['natural_movie_one.npy', 'natural_movie_two.npy', 'natural_movie_three.npy'],#, 'natural_natural_scenes.npy'],
    'clip': ['natural_movie_one_clip.npy', 'natural_movie_two_clip.npy', 'natural_movie_three_clip.npy'],#, 'natural_natural_scenes_clip.npy']
    'dino': ['natural_movie_one_dino.npy', 'natural_movie_two_dino.npy', 'natural_movie_three_dino.npy'],# 'natural_natural_scenes_dino.npy']
    #'vitmae': ['natural_movie_one_vitmae.npy', 'natural_movie_two_vitmae.npy', 'natural_movie_three_vitmae.npy'],
    'vit': ['natural_movie_one_vit.npy', 'natural_movie_two_vit.npy', 'natural_movie_three_vit.npy']
}

stimuli_dct = {
    'natural_movie_one': {'clip': 'natural_movie_one_clip.npy',
         'dino': 'natural_movie_one_dino.npy',
         #'vitmae': 'natural_movie_one_vitmae.npy',
         'vit': 'natural_movie_one_vit.npy'}
}

#https://huggingface.co/google/vit-base-patch32-384

stimulus_session_dict= {
    'three_session_A': ['natural_movie_one', 'natural_movie_three'],
    'three_session_B': ['natural_movie_one'],
    'three_session_C': ['natural_movie_one', 'natural_movie_three'],
    'three_session_C2': ['natural_movie_one', 'natural_movie_three']
}