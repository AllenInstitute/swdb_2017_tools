import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import os
from allensdk.brain_observatory.natural_scenes import NaturalScenes
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'
manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
targeted_structures=['VISrl','VISp']
cre_lines=['Cux2-CreERT2']
imaging_depths=[175]
stimuli=['natural_scenes']



boc=BrainObservatoryCache(manifest_file=manifest_file)
def get_experiment_ids(boc,targeted_structure,cre_line,imaging_depth):
    '''Gives back session IDs for an experiment
    specified by targeted structure, cre-line, and imaging depth'''
    exps = boc.get_experiment_containers(targeted_structures=targeted_structure,
                                       cre_lines=cre_line,imaging_depths=imaging_depth)
    ids = [exp['id'] for exp in exps]
    return ids

def get_rsms(boc, drive_path, targeted_structures,cre_lines,imaging_depths,stimuli):
    ''' Gives back rsms

        Parameters
        ----------
        boc : object
        stimuli : string
            'drifting_gratings', 'locally_sparse_noise', 'locally_sparse_noise_4deg',
            'locally_sparse_noise_8deg', 'natural_movie_one', 'natural_movie_three', 'natural_movie_two',
            'natural_scenes', 'spontaneous', or 'static_gratings'
        drive_path : string
            path to the Brain Observatory Cache
        ids: list
            List of experiment container ids.
        targeted_structures: list
            List of structure acronyms.  Must be in the list returned by
            BrainObservatoryCache.get_all_targeted_structures().
        imaging_depths: list
            List of imaging depths.  Must be in the list returned by
            BrainObservatoryCache.get_all_imaging_depths().
        cre_lines: list
            List of cre lines.  Must be in the list returned by
            BrainObservatoryCache.get_all_cre_lines().'''
    rsms = []
    experiment_ids=get_experiment_ids(boc,targeted_structures,cre_lines,imaging_depths) #self.experiment_ids
    for experiment_id in experiment_ids:
        exps = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=stimuli)
        session_id = exps[0]['id']
        data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
        f.close()
        if stimuli[0] == 'natural_scenes':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_B_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_ns'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0] == 'static_gratings':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_A_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_sg'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0] == 'drifting_gratings':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_A_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_dg'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0]=='natural_movies':
            #for later
            print 'sorry not yet'
        else:
            raise Exception('stimulus not found')
    return rsms



def vectorize(mat):
    """Takes a square symmetric matrix mat and returns thse vectorized form. Like matlab's squareform.

         Note: could probably also just use scipy's squareform function."""
    assert mat.shape[0] == mat.shape[1]

    vec = mat[:, 0]
    for row in range(1, mat.shape[1]):
        vec = np.concatenate((vec, mat[row:, row]))

    return vec


def get_rsm_corr():
    ''' Function that gets Kendall-tau-a measurements between all RSM matrices '''
    print 'yay'

def get_kt(rsm1,rsm2):
    '''Gets Kendall tau-a measurements between two RSM matrices, first vectorizes matrices
    and then computes kt using scipy kendall-tau function

    Parameters
    -------------------------------
    rsm1 : response data matrix of size n x n

    rsm2 : response data matrix of size n x n

    '''
    #vecRDM1 = vectorize(RDM1)
    #vecRDM2 = vectorize(RDM2)
    vec_rsm1 = scipy.spatial.distance.squareform(rdm1)
    vec_rsm1 = scipy.spatial.distance.squareform(rdm2)
    k = kt(vec_rsm1, vec_rsm1).correlation
    return k

#kt=get_kt(rsm1.rsm,rsm2.rsm)
#classdef rsm(object):
 #   '''Create response similarity matrix '''
 #   __init__