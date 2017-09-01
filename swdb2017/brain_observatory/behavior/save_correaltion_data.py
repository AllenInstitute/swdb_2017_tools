import os
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import h5py
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
%matplotlib inline
from tqdm import tqdm_notebook

from swdb2017.brain_observatory.behavior.get_pupil_size_data import has_pupil_data_df
import swdb2017.brain_observatory.behavior.extract_pupil_features as epf
import swdb2017.brain_observatory.behavior.extract_running_features as erf
import swdb2017.brain_observatory.behavior.correlate_fluor_behavior as cfb
import swdb2017.brain_observatory.behavior.correlation_matrix as cm

from trash_cache import TrashCache
tc = TrashCache(manifest_fp='/home/fionag/tpc/trash_cache_manifest.json')

def save_behavior_correlations(boc, tc, experiment_ids=[], features=[], figure=False):
    for i in tqdm_notebook(experiment_ids):
        # Load data
        exp_data = boc.get_ophys_experiment_data(i)
        exp_meta = exp_data.get_metadata()

        # Calculate correlations
        if exp_meta['session_type'] == 'three_session_B':
            ns_behavior_corr_df = cfb.corr_ns_behavior(exp_data, raw=False)
        else:
            ns_behavior_corr_df = ['no_data']

        spont_behavior_corr_df = cfb.corr_spont_behavior(exp_data, raw=False)
        behavior_behavior_corr_df = cm.get_correlations_from_features(boc, features, exp_id=i, figure=figure)

        # Save data
        tc.save_experiments([{'id': i, 'vars': [{'name': 'ns_behavior_correlation', 'data': ns_behavior_corr_df},
                                                {'name': 'spont_behavior_correlation', 'data': spont_behavior_corr_df},
                                                {'name': 'behavior_behavior_correlation',
                                                 'data': behavior_behavior_corr_df},
                                                {'name': 'metadata', 'data': exp_meta}]
                              }])