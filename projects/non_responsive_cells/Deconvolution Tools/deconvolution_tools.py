import numpy as np
import pandas as pd
import os
import sys
import h5py


def get_spiking_data(dff_traces, timestamps, sig=3):
    from OASIS.functions import deconvolve

    spike_times = []
    isis = []
    spikes = []

    for i, dff_trace in enumerate(dff_traces):
        c, s, b, g, lam = deconvolve(np.double(dff_trace), penalty=1)
        s_sig = (s >= sig * np.std(s))
        spikes.append(s_sig * 1)
        spike_times.append(timestamps[s_sig])
        isis.append(np.diff(spike_times[-1]))

    spike_data = {}
    spike_data['isis'] = isis
    spike_data['spikes'] = spikes
    spike_data['spike_times'] = spike_times

    return spike_data


def get_dff(boc, expt_container_id, ophys_experiment_id):
    expt_session_info = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id])
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    dff_traces = dataset.get_dff_traces()[1]
    timestamps = dataset.get_fluorescence_timestamps()

    return dff_traces, timestamps


def plot_raster(spike_times):
    import matplotlib.pyplot as plt

    plt.figure(num=None, figsize=(16, 6), dpi=80, facecolor='w', edgecolor='k')
    for i, spike_time in enumerate(spike_times):
        plt.scatter(spike_time, np.ones(len(spike_time)) + i, s=0.2, c='k', marker='o')
    plt.xlabel('Time(s)')
    plt.ylabel('Cell Number')
    plt.show()
