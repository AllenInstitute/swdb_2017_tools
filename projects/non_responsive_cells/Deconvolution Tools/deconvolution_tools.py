import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def get_spiking_data(dff_traces, timestamps, sig=3)

    spike_times = []
    isis = []
    spikes = []

    for i, dff_trace in enumerate(dff_traces):
        c, s, b, g, lam = deconvolve(np.double(dff_trace), penalty=1)
        s_sig = (s >= sig * np.std(s))
        spikes.append(s_sig * 1)
        spike_times.append(timestamps[s_sig])
        isis.append(np.diff(spike_times[-1]))

    data['isis'] = isis
    data['spikes'] = spikes
    data['spike_times'] = spike_times

    return data


def get_allen_data(boc, expt_container_id, ophys_experiment_id):

    expt_session_info = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id])
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=ophys_experiment_id)
    dff_traces = dataset.get_dff_traces()[1]
    timestamps = dataset.get_fluorescence_timestamps()

    return dff_traces, timestamps


def plot_raster(spike_times)
