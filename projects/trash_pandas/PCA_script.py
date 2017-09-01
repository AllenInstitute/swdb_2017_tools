'''
Outer wrapper script for PCA analysis. This script calls PCA_batch function
(located in PCA_Behavior_Analysis.py) which completes the analysis and returns a
dictionary of analysis results. These results are then saved using the trash_cache

Instructions:
Change the variable "stim_type" in order to choose which class of images you'd like
to analyze. If 'natural_scenes' is specified, provide a list of the natural_scenes
you'd like to analyze. These results will all be saved into directories corresponding
to their ids within the trash_cache located on external hard drive
'''


# Set drive path to the brain observatory cache located on hard drive
drive_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'

# Import standard libs
import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt
import load_by_stim as lbs
import plotting as c_plt
from matplotlib.collections import LineCollection
import scipy.ndimage.filters as filt
from PCA_Behavior_Analysis import PCA_batch
from trash_cache import TrashCache
import extract_pupil_features as epf
import extract_running_features as err

# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)


# Get list of all stimuli
stim = boc.get_all_stimuli()
structures = boc.get_all_targeted_structures()
cre_lines = boc.get_all_cre_lines()


# Get all ophys experiments with eye tracking data, for spont period
exps_ = boc.get_ophys_experiments(stimuli = ['natural_scenes'], simple = False, targeted_structures=structures)
exps = []

stim_type = 'natural_scenes'
images = [1, 10, 20, 30, 40, 50]

for i, exp in enumerate(exps_):
    if (exps_[i]['fail_eye_tracking']==False):
        exps.append(exps_[i])

## Iterate over all experiments and cache analysis
for i in range(0, len(exps)):
    exp_id = exps[i]['id']
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id = exp_id)
    cell_ids = data_set.get_cell_specimen_ids()
    meta_data = data_set.get_metadata()
    if len(cell_ids) <= 20:    # Get rid of experiments with less than 20 cells
        continue
    if stim_type == 'spont':
        trash_cache_path = '/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/spontPCA'
        tc = TrashCache(os.path.join(trash_cache_path, 'trash_cache_manifest.json'))
        pca, behavior = PCA_batch(data_set, stim_type = 'spont')

        exp = dict(id=exp_id, vars = [dict(name='var_explained', data = pca['var_explained']),
        dict(name='corr_mat', data=behavior['corr_mat'].T),
        dict(name='PCs', data = pca['PCs']),
        dict(name='Principle axes', data = pca['axes']),
        dict(name='Eigenvalues', data = pca['weights']),
        dict(name='Fraction of PCs', data = pca['fraction_pcs']),
        dict(name='Experiment data', data = behavior['data']),
        dict(name ='meta_data', data = meta_data)
        ]
        )
        tc.save_experiments([exp])

    elif stim_type == 'natural_scenes':

        for image in images:
            trash_cache_path = os.path.join('/media/charlie/Brain2017/data/dynamic-brain-workshop/trash_cache/nsPCA', str(image))
            tc = TrashCache(os.path.join(trash_cache_path, 'trash_cache_manifest.json'))
            pca, behavior = PCA_batch(data_set, stim_type = 'natural_scenes', images = image)

            exp = dict(id=exp_id, vars = [dict(name='var_explained', data = pca['var_explained']),
            dict(name='corr_mat', data=behavior['corr_mat'].T),
            dict(name='PCs', data = pca['PCs']),
            dict(name='Principle axes', data = pca['axes']),
            dict(name='Eigenvalues', data = pca['weights']),
            dict(name='Fraction of PCs', data = pca['fraction_pcs']),
            dict(name='Experiment data', data = behavior['data']),
            dict(name ='meta_data', data = meta_data)
            ]
            )

            tc.save_experiments([exp])
