from pathlib import Path
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import numpy as np

# Get data
# We want to get all the experiments with one cre line-- that's round enough for us
# We want to extract two train and validation sets-- select some number of unseen images
# We need extraction to give us data for movie one and natural images-- We are only targeting
# experiment B.
# Run regression experiments. Get data for all selected cell specimen and put them
# in the corresponding df.

output_dir = '/media/maria/DATA/AllenData'

# Get data


class MakeTestTrain():
    def __init__(self, cre_line):
        self.cre_line = cre_line
        self.output_dir = '/media/maria/DATA/AllenData'
        self.boc = BrainObservatoryCache(manifest_file=str(
            Path(output_dir) / 'brain_observatory_manifest.json'))

    def _get_eids(self):
        self.experiment_container = self.boc.get_experiment_containers(cre_lines=[
                                                                       self.cre_line])
        eids = []
        z = 0
        for e in self.experiment_container:
            if z < 4:
                id = e['id']
                exps = self.boc.get_ophys_experiments(
                    experiment_container_ids=[id])
                for i in exps:
                    if i['session_type'] == 'three_session_B':
                        eids.append(i['id'])
            z += 1
        print(eids)
        return eids

    def fit_transform(self):
        eids = self._get_eids()
        data_dct = {}
        for eid in eids:
            data_dct[eid] = {}
        for eid in eids:
            data_set = self.boc.get_ophys_experiment_data(eid)
            movie_stim_table = data_set.get_stimulus_table('natural_movie_one')
            data_dct[eid]['movie_stim'] = data_set.get_stimulus_template(
                'natural_movie_one')
            train_stimulus_table = data_set.get_stimulus_table(
                'natural_scenes')
            data_dct[eid]['natural_stim'] = data_set.get_stimulus_template(
                'natural_scenes')
        '''
        #Check whether stimulus templates are the same
        for k in data_dct.keys():
            for l in data_dct.keys():
                if k != l:
                    if np.array_equal(data_dct[k]['movie_stim'], data_dct[l]['movie_stim']):
                        print(True)
                    else:
                        print(False)
                    if np.array_equal(data_dct[k]['natural_stim'], data_dct[l]['natural_stim']):
                        print(True)
                    else:
                        print(False)
        '''


MakeTestTrain('Emx1-IRES-Cre').fit_transform()
