import numpy as np
import pandas as pd
import os
import sys
import h5py


def get_spiking_data(dff_traces, timestamps, cell_specimen_ids, num_std=3):
    '''
    Deconvolve dff_traces into spikes
    uses OASIS (https://github.com/j-friedrich/OASIS)

    Inputs:
        dff_traces: dictoionary dff_traces
        timestamps: time vector of dff_traces
        std: standard deviation required for classifying as a spike
    Returns:
        isis - dictionary
            key: cell_specimen_id, value:  interspike intervals
        spikes - dictionary
             key: cell_specimen_id, value:  binary vector indicating where spikes were detected
        spiketimes - dictionary
            key: cell_specimen_id, value:  spiketimes
        timestamps: list containing dff_timestamps
    '''

    from oasis.functions import deconvolve

    dff_trace = dff_traces[0]
    spike_prob_list = []
    dff = {}

    for i, dff_trace in enumerate(dff_traces):
        c, s, b, g, lam = deconvolve(np.double(dff_trace), penalty=1)
        spike_prob_list.append(s)
        dff[cell_specimen_ids[i]] = dff_trace

    std = np.std(np.asarray(spike_prob_list))

    spike_times = {}
    isis = {}
    spikes = {}

    for i, spike_prob in enumerate(spike_prob_list):
        s_sig = (spike_prob >= num_std * std)
        spikes[cell_specimen_ids[i]] = s_sig * 1
        spike_times[cell_specimen_ids[i]] = timestamps[s_sig]
        isis[cell_specimen_ids[i]] = np.diff(spike_times[cell_specimen_ids[i]])

    return spikes, spike_times, isis


def get_dff(boc, ophys_experiment_id):
    '''
    Get entire dff_traces for a given ophys_experiment

    Inputs:
        boc: BrainObservatoryCache Object
        ophys_experiment_id
    Returns:
        dff_traces: List of dff_traces
        timestamps: time vector for dff_traces
        cell_specimen_ids: cell_ids
    '''
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    timestamps, dff_traces = dataset.get_dff_traces()
    cell_specimen_ids = dataset.get_cell_specimen_ids()

    return dff_traces, timestamps, cell_specimen_ids


def plot_raster(spike_times, title=' '):
    '''
    Create a raster plot based on spikes_times

    input:
        spike_times: A list of lists of spike times

    To do:
        Change so it accomidates dictionaries
    '''
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')

    if isinstance(spike_times, dict):
        i = 0
        for k, spike_time in spike_times.iteritems():
            plt.scatter(spike_time, np.ones(len(spike_time)) + i, s=0.2, c='k', marker='o')
            i+=1

    if isinstance(spike_times, list):
        for i, spike_time in enumerate(spike_times):
            plt.scatter(spike_time, np.ones(len(spike_time)) + i, s=0.2, c='k', marker='o')

    plt.xlabel('Time(s)')
    plt.ylabel('Cell Number')
    plt.title(title)
    plt.show()


def get_all_stimulus_epoch_data(boc, ophys_experiment_id, plot=0):
    '''
    Get dff, timestamps, and spiking data for each stimulus epoch

    Inputs:
        boc: BrainObservatoryCache Object
        ophys_experiment_id: ophys_experiment_id
    Returns:
        data: Dictionary containing
            key: stimulus, values dictionaries
                dff -  key: cell_specimen_id, value: dff_trace
                timestamps -  key: cell_specimen_id, value: timestamps for dff_trace
                spikes -  key: cell_specimen_id, value: spike locations in dff_trace
                spike_times -  key: cell_specimen_id, value: spike times
                isis -  key: cell_specimen_id, value: isis
    '''
    dff_traces, timestamps, cell_specimen_ids = get_dff(boc=boc, ophys_experiment_id=ophys_experiment_id)

    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    stimulus_epoch_tables = dataset.get_stimulus_epoch_table()

    data = {}
    dff_traces = np.asarray(dff_traces)

    for index, row in stimulus_epoch_tables.iterrows():
        stim = row['stimulus']
        start = row['start']
        end = row['end']

        data[stim] = {}

        data[stim]['dff'] = dff_traces[:, start:end].tolist()
        data[stim]['timestamps'] = timestamps[start:end]

        data[stim]['spikes'], data[stim]['spike_times'], data[stim]['isis'] = get_spiking_data(
            data[stim]['dff'], data[stim]['timestamps'], cell_specimen_ids, 7)

        if plot != 0:
            data_to_plot = [data[stim]['spike_times'][cells] for cells in data[stim]['spike_times']]
            plot_raster(spike_times=data_to_plot, title=stim)

    return data


def get_dff_epoch_data(boc, ophys_experiment_id):

    dff_traces, timestamps, cell_specimen_ids = get_dff(boc=boc, ophys_experiment_id=ophys_experiment_id,)
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    stimulus_epoch_tables = dataset.get_stimulus_epoch_table()
    for index, row in stimulus_epoch_tables.iterrows():
        stim = row['stimulus']
        start = row['start']
        end = row['end']
        data[stim]['dff'] = dff_traces[:, start:end].tolist()
        data[stim]['timestamps'] = timestamps[start:end]
    return data


def get_epoch_spiking_data(boc, ophys_experiment_id, num_std=6):
    '''
    get spiking data for an ophys_experiment

    Inputs:
        boc: BrainObservatoryCache
        ophys_experiment_id: ophys_experiment_id
        std: standard deviation used to determine spike threshold from the
             output of the deconvolution algorithm
     Returns:
        spikes: keys: cell_specimen_ids values: lists indicating spike occurances
        timestamps_dict: lell_specimen_ids values: list of timestamps
        spike_times: keys: cell_specimen_ids values: spike time
        isis: keys: cell_specimen_ids values: interspike intervals
    '''
    dff_traces, timestamps, cell_specimen_ids = get_dff(boc=boc, ophys_experiment_id=ophys_experiment_id)

    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    stimulus_epoch_tables = dataset.get_stimulus_epoch_table()

    spikes = {}
    spike_times = {}
    isis = {}
    data = {}
    timestamps_dict = {}

    dff_traces = np.asarray(dff_traces)

    for index, row in stimulus_epoch_tables.iterrows():
        stim = row['stimulus']
        start = row['start']
        end = row['end']

        data[stim] = {}

        spikes[stim] = {}
        spike_times[stim] = {}
        isis[stim] = {}

        data[stim]['dff'] = dff_traces[:, start:end].tolist()
        timestamps_dict[stim] = timestamps[start:end]

        spikes[stim], spike_times[stim], isis[stim] = get_spiking_data(
            data[stim]['dff'], timestamps_dict[stim], cell_specimen_ids, num_std=num_std)

    return spikes, timestamps_dict, spike_times, isis
