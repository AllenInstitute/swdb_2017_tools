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
from pearson_corr_coeff import pearson_corr_coeff
from swdb2017.brain_observatory.util.cell_specimen_ops import get_run_mod_cells
# Import brain observatory cache class. This is responsible for downloading any data
# or metadata

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

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
meta_data = data_set.get_metadata()

############ Load and pre-process data for dim-reduction #####################

# get df/f traces for spont activity
out, spont_cell_ids = lbs.get_spont_specific_fluorescence_traces(data_set, False, binned=False)
nTrials = len(out['fluorescence']['spont'])
dff_spont = np.squeeze(np.stack(out['fluorescence']['spont'][0:nTrials],axis=1))
pr_spont = np.concatenate(out['pupil size']['spont'][0:nTrials], axis = 0)
f_mean = np.nanmean(dff_spont, axis =1)
f_std = np.nanmean(dff_spont, axis=1)
for i in range(0, len(f_mean)):
    dff_spont[i,:] = (dff_spont[i,:] - f_mean[i])/f_std[i]
pr_spont = (pr_spont - np.nanmean(pr_spont))/np.nanstd(pr_spont)


# get df/f traces for natural scences
image = 5
binned = False
out, ns_cell_ids = lbs.get_ns_specific_fluorescence_traces(data_set, False, binned = binned)
nTrials = len(out['fluorescence'][0])

if binned:
    pr_ns = out['pupil size'][image][0:nTrials]
    dff_ns = np.stack(out['fluorescence'][image][0:nTrials], axis=1)
else:
    pr_ns = np.concatenate(out['pupil size'][image][0:nTrials], axis=0)
    dff_ns = np.concatenate(out['fluorescence'][image][0:nTrials], axis=1)
f_mean = np.nanmean(dff_ns, axis =1)
f_std = np.nanmean(dff_ns, axis=1)
for i in range(0, len(f_mean)):
    dff_ns[i,:] = (dff_ns[i,:] - f_mean[i])/f_std[i]
pr_ns = (pr_ns - np.nanmean(pr_ns))/np.nanstd(pr_ns)


# Qualitative look at spont population acitivity
plt.figure()
plt.title('Spontaneous activity')
plt.imshow(dff_spont, aspect='auto')
plt.xlabel('frames')
plt.ylabel('cells')


# Qualitative look at ns population acitivity
plt.figure()
plt.title('Natural scene "0" responses')
plt.imshow(dff_ns, aspect='auto')
plt.xlabel('frames')
plt.ylabel('cells')


######## Singular value decomposition of NS responses for given stim ##########
U, S, V = np.linalg.svd(dff_ns)
corr = np.zeros(len(S))
for i in range(0, len(S)):
    corr[i] = pearson_corr_coeff(V.T[i], pr_ns)

pcs = U[:,0:len(S)]*S
plt.figure()
plt.subplot(211)
plt.plot(V.T[0])
plt.subplot(212)
plt.plot(V[0])
# Make single trial projections
'''
Vt = V.transpose()
l = Vt.shape[1]
trialLen = 7
V1 = filt.gaussian_filter(V[0].reshape(trialLen, l/trialLen),.5)
V1mean = np.mean(V1,axis=0)
V2 = filt.gaussian_filter(V[1].reshape(trialLen, l/trialLen),.5)
V2mean = np.mean(V2,axis=0)
'''
var_explained = []
for i in range(0, len(S)):
    var_explained.append(sum(S[0:i])/sum(S))
fig, ax = plt.subplots(2, 2)
fig.suptitle('PCA ns')
ax[0,0].set_ylabel('Fraction of variance explained')
ax[0,0].set_xlabel('n principle components')
ax[0,0].plot(var_explained, '-o')
ax[0,0].plot(0.5*np.ones(len(S)), '--r')
ax[1,0].plot(pcs[0], pcs[1], '.')
ax[1,0].set_xlabel('PC1')
ax[1,0].set_ylabel('PC2')
ax[1,1].plot(pr_ns)
ax[0,1].plot(corr)
cm = plt.get_cmap('jet')
'''
for trial in range(0, V1.shape[1]):
    x = c_plt.colorline(V1[:,trial], V2[:,trial], cmap=plt.get_cmap('copper'), linewidth=1, alpha = 0.2)
    ax[0,1].add_collection(x)
x = c_plt.colorline(V1mean, V2mean, cmap=plt.get_cmap('copper'), linewidth=3, alpha = 1)
ax[0,1].add_collection(x)
ax[0,1].autoscale(True)
'''

############ Singular value decomposition of spont acitivity ###################
U, S, V = np.linalg.svd(dff_spont)
corr = np.zeros(len(S))
for i in range(0, len(S)):
    corr[i] = pearson_corr_coeff(V.T[i], pr_spont)

pcs = U*S
var_explained = []
for i in range(0, len(S)):
    var_explained.append(sum(S[0:i])/sum(S))
fig, ax = plt.subplots(2, 2)
fig.suptitle('PCA spont')
ax[0,0].set_ylabel('Fraction of variance explained')
ax[0,0].set_xlabel('n principle components')
ax[0,0].plot(var_explained, '-o')
ax[0,0].plot(0.5*np.ones(len(S)), '--r')
ax[1,0].plot(pcs[0], pcs[1], '.')
ax[1,0].set_xlabel('PC1')
ax[1,0].set_ylabel('PC2')
ax[1,1].plot(pr_spont)
ax[0,1].plot(corr)
plt.show()
