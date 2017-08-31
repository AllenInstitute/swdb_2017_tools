# Import standard libs
import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt

from allensdk.brain_observatory.natural_scenes import NaturalScenes


def get_pupil_mean_sweep_response(data_set, analysis_file, stim_type):
    '''
    Arguments:
    --------------------------------------------------------------------------
    data_set: A specific experiment from get_ophys_experiment_data
    analysis_file: corresponding analysis file that holds the mean sweep response for this stim
    stim_type: str, Stimulus category i.e. 'natural_scenes'

    Returns:
    --------------------------------------------------------------------------
    msr: data frame of mean sweep responses with a column added containing mean pupil size
    threshold: mean pupil size over total experiment
    header: str, column name for the pupil mean
    '''


    analysis_objects = {
        'natural_scenes': NaturalScenes
    }
    ns = analysis_objects[stim_type].from_analysis_file(data_set,analysis_file)
    msr = ns.mean_sweep_response
    del msr['dx']   # get rid of uneeded comlumn
    t, pr = data_set.get_pupil_size()
    stim_table = data_set.get_stimulus_table('natural_scenes')
    threshold = np.mean(pr)
    pm = np.zeros(stim_table.shape[0])

    for i in np.arange(0, stim_table.shape[0]):
        start = stim_table['start'][i]
        end = stim_table['end'][i] + 15    # Add 500ms
        pm[i] = np.nanmean(pr[start:end])

    header = 'pupil'
    msr[header] = pm

    return msr, threshold, header
