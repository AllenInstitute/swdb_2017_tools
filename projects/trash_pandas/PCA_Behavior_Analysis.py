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


import extract_pupil_features as epf
import extract_running_features as err

# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache





import numpy as np
import pandas as pd
from swdb2017.brain_observatory.utilities.z_score import z_score
from swdb2017.brain_observatory.behavior.correlation_matrix import pearson_corr_coeff
def PCA_batch(data_set, smoothed=0):

    '''
    Uses singular value decomposition to decompose population activity into
    its principle components for a given stimulus. Runs correlation analysis
    between eigenvectors of population acitivty and behavioral correlates.
    Will z-score the input data.

    Args:
    ------------------------------
    data_set: an instance of get_ophys_experiment_data for a given experiment id
    stim: stimulus category i.e. "natural_scenes"  (will just use spont for now)


    Returns:
    ------------------------------
    PCA: dict
        PCs: contains principle components
        Weights: contains eigenvalues
        Axes: contains eigenvectors
        var_explained: net variance explained sorted by PC number

    Behavior: dict
        corr_mat: data frame containing correlation between each principle axes and
        each time varying behavioral readout
        data: data frame containing each of the behavioral readouts that were
        used in corr_mat in addition to the df/F trace for each cell
    '''
    # load the df/F trace for the spont input experiment
    out, spont_cell_ids = lbs.get_spont_specific_fluorescence_traces(data_set, False, binned=False)
    nTrials = len(out['fluorescence']['spont'])
    nCells = len(spont_cell_ids)
    print(nTrials)
    # Concatenate all spont periods into a numpy matrix for use with PCA and
    # correlation analysis. Do this for df/F and for all behavioral readouts
    behavior_df = pd.DataFrame([], index = [0], columns = out.keys())
    for key in out.keys():
        if key == 'fluorescence':
            behavior_df[key][0] = z_score((np.concatenate(out['fluorescence']['spont'][0:nTrials], axis=1)))
        elif key == 'pupil location':
            behavior_df[key][0] = []
        else:
            behavior_df[key][0] = (np.concatenate(out[key]['spont'][0:nTrials].values, axis=0))

    # Perform SVD on fluorescence traces
    U, S, V = np.linalg.svd(behavior_df['fluorescence'][0])
    pcs = U*S
    axes = V.T
    weights = S
    var_explained = []
    for i in range(0, len(S)):
        var_explained.append(sum(S[0:i])/sum(S))

    # perform correlation for each behavioral readout with each princile axes
    corr_mat = pd.DataFrame([], index = np.arange(0,nCells), columns = out.keys())
    for i in np.arange(0, nCells):
        for key in out.keys():
            if key == 'fluorescence' or key == 'pupil location':
                break
            else:
                corr_mat[key][i] = pearson_corr_coeff(axes[i], behavior_df[key][0])


    return behavior_df, corr_mat









manifest_file = os.path.join(drive_path, 'brain_observatory_manifest.json')

boc = BrainObservatoryCache(manifest_file=manifest_file)

# Get list of all stimuli
stim = boc.get_all_stimuli()
# Select brain region of interest and cre lines
targeted_structures = ['VISp']
cre_line = 'Rbp4-Cre_KL100'

# Get all ophys experiments with eye tracking data, for spont period
exps_ = boc.get_ophys_experiments(stimuli = ['natural_scenes'], simple = False, targeted_structures=targeted_structures)
exps = []
for i, exp in enumerate(exps_):
    if (exps_[i]['fail_eye_tracking']==False):
        exps.append(exps_[i])

## Test PCA with one of the experiments in exps
exp_id = exps[6]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id = exp_id)
meta_data = data_set.get_metadata

bdf, corr = PCA_batch(data_set)
print(corr)
plt.figure()
plt.imshow(corr)
plt.show()
