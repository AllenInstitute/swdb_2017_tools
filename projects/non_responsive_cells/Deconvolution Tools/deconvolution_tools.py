import numpy as np
import pandas as pd
import os
import sys
import h5py


def get_spiking_data(dff_traces, timestamps, cell_specimen_ids, sig=3):
    '''
    Deconvolve dff_traces into spikes
    uses OASIS (https://github.com/j-friedrich/OASIS)

    Inputs:
        dff_traces: dictoionary dff_traces
        timestamps: time vector of dff_traces
        sig: standard deviation required for classifying as a spike
    Returns:
        spike_data: Dictionary containing
            Key: isis, Value: Dictionary containing
                key: cell_specimen_id, value:  interspike intervals
            spikes: Dictionary containing
                 key: cell_specimen_id, value:  binary vector indicating where spikes were detected
            KeyL spike_times: Dictionary containing
                key: cell_specimen_id, value:  spiketimes
            Key: timestamps: dff_timestamps
    '''

    from OASIS.functions import deconvolve

    spike_times = {}
    isis = {}
    spikes = {}

    for i, dff_trace in enumerate(dff_traces):
        c, s, b, g, lam = deconvolve(np.double(dff_trace), penalty=1)
        s_sig = (s >= sig * np.std(s))
        spikes[cell_specimen_ids[i]] = s_sig * 1
        spike_times[cell_specimen_ids[i]] = timestamps[s_sig]
        isis[cell_specimen_ids[i]] = np.diff(spike_times[cell_specimen_ids[i]])



    spike_data = {}
    spike_data['isis'] = isis
    spike_data['spikes'] = spikes
    spike_data['spike_times'] = spike_times
    spike_data['timestamps'] = timestamps

    return spike_data


def get_dff(boc, ophys_experiment_id):
    '''
    Get dff_traces for a given ophys_experiment

    Inputs:
        boc: BrainObservatoryCache Object
        ophys_experiment_id
    Returns:
        dff_traces: List of dff_traces
        timestamps: time vector for dff_traces
        cell_specimen_ids: cell_ids
    '''
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    dff_traces = dataset.get_dff_traces()[1]
    timestamps = dataset.get_fluorescence_timestamps()
    cell_ids = dataset.get_cell_specimen_ids()

    return dff_traces, timestamps, cell_specimen_ids


def plot_raster(spike_times):
    '''
    Create a raster plot based on spikes_times

    input:
        spike_times: A list of lists of spike times
    '''
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    for i, spike_time in enumerate(spike_times):
        plt.scatter(spike_time, np.ones(len(spike_time)) + i, s=0.2, c='k', marker='o')
    plt.xlabel('Time(s)')
    plt.ylabel('Cell Number')
    plt.show()
