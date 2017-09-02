import numpy as np
import pandas as pd
import load_by_stim as lbs
from swdb2017.brain_observatory.utilities.z_score import z_score
from swdb2017.brain_observatory.behavior.correlation_matrix import pearson_corr_coeff

def PCA_batch(data_set, stim_type, images = 0):
    '''
    Uses singular value decomposition to decompose population activity into
    its principle components for a given stimulus. Runs correlation analysis
    between eigenvectors of population acitivty and behavioral correlates.
    Will z-score the input data.

    Args:
    ------------------------------
    data_set: an instance of get_ophys_experiment_data for a given experiment id
    stim: stimulus category i.e. "natural_scenes"  (will just use spont for now)
    stim_type: string, 'natural_scenes' or 'spont'
    images: optional, if given, is list of natural scenes to analyze

    Returns:
    ------------------------------
    PCA: dict
        PCs: contains principle components
        Weights: contains eigenvalues
        Axes: contains eigenvectors
        var_explained: net variance explained sorted by PC number
        fraction_pcs: fraction of pcs needed to explain 50 per of variance

    Behavior: dict
        corr_mat: data frame containing correlation between each principle axes and
        each time varying behavioral readout
        data: data frame containing each of the behavioral readouts that were
        used in corr_mat in addition to the df/F trace for each cell
    '''

# -------------------------- Spont acitivty ----------------------------------

    if stim_type == 'spont':
        # load the df/F trace for the spont input experiment
        out, spont_cell_ids = lbs.get_spont_specific_fluorescence_traces(data_set, False, binned=False)
        nTrials = len(out['fluorescence']['spont'])
        nCells = len(spont_cell_ids)

        # Concatenate all spont periods into a numpy matrix for use with PCA and
        # correlation analysis. Do this for df/F and for all behavioral readouts

        out.pop('pupil location', None)
        keys = out.keys()
        behavior_df = pd.DataFrame([], index = [0], columns = keys)
        for key in out.keys():
            if key == 'fluorescence':
                behavior_df[key][0] = z_score((np.concatenate(out['fluorescence']['spont'][0:nTrials], axis=1)))
            else:
                behavior_df[key][0] = (np.concatenate(out[key]['spont'][0:nTrials].values, axis=0))

        # Perform SVD on fluorescence traces
        U, S, V = np.linalg.svd(behavior_df['fluorescence'][0])
        pcs = U*S
        axes = V.T
        weights = S
        nPcs = nCells
        var_explained = np.empty(nPcs)
        for i in range(0, len(S)):
            var_explained[i]=(sum(S[0:i+1])/sum(S))

        # perform correlation for each behavioral readout with each princile axes
        # gets rid of readouts that don't matter
        l = ['fluorescence', 'time', 'pupil location']
        for ids in l:
            out.pop(ids, None)

        fraction = float(np.where(var_explained > 0.5)[0][0])/float(nPcs)
        columns = np.sort(out.keys())
        corr_mat = pd.DataFrame([], index = np.arange(0,nPcs), columns = columns)
        for i in np.arange(0, nPcs):
            for key in columns:
                if key != 'fluorescence':
                    corr_mat[key][i] = pearson_corr_coeff(axes[i], behavior_df[key][0])
        corr_mat = corr_mat[corr_mat.columns].astype(float)

# ---------------------- Natural Scenes --------------------------------------

    elif stim_type == 'natural_scenes':
        # load the df/F trace for the ns input experiment
        out, spont_cell_ids = lbs.get_ns_specific_fluorescence_traces(data_set, False, binned=False)
        image = images
        nTrials = len(out['fluorescence'][image])
        nCells = len(spont_cell_ids)

        # Concatenate all spont periods into a numpy matrix for use with PCA and
        # correlation analysis. Do this for df/F and for all behavioral readouts

        out.pop('pupil location', None)
        keys = out.keys()
        behavior_df = pd.DataFrame([], index = [0], columns = keys)
        for key in out.keys():
            if key == 'fluorescence':
                behavior_df[key][0] = z_score((np.concatenate(out['fluorescence'][image][0:nTrials], axis=1)))
            else:
                behavior_df[key][0] = (np.concatenate(out[key][image][0:nTrials].values, axis=0))

        # Perform SVD on fluorescence traces
        U, S, V = np.linalg.svd(behavior_df['fluorescence'][0])
        pcs = U*S
        axes = V.T
        weights = S
        nPcs = nCells
        var_explained = np.empty(nPcs)
        for i in range(0, len(S)):
            var_explained[i]=(sum(S[0:i+1])/sum(S))

        # perform correlation for each behavioral readout with each princile axes
        # gets rid of readouts that don't matter
        l = ['fluorescence', 'time', 'pupil location']
        for ids in l:
            out.pop(ids, None)

        fraction = float(np.where(var_explained > 0.5)[0][0])/float(nPcs)
        columns = np.sort(out.keys())
        corr_mat = pd.DataFrame([], index = np.arange(0,nPcs), columns = columns)
        for i in np.arange(0, nPcs):
            for key in columns:
                if key != 'fluorescence':
                    corr_mat[key][i] = pearson_corr_coeff(axes[i], behavior_df[key][0])
        corr_mat = corr_mat[corr_mat.columns].astype(float)


    PCA = dict()
    PCA['PCs'] = pcs
    PCA['axes'] = axes
    PCA['weights'] = weights
    PCA['var_explained'] = var_explained
    PCA['fraction_pcs'] = fraction
    Behavior = dict()
    Behavior['data'] = behavior_df
    Behavior['corr_mat'] = corr_mat
    return PCA, Behavior
