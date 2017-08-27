import numpy as np
import pandas as pd


def get_spont_specific_fluorescence_traces(exp, raw):
    # Arguments:
    # exp: Individual experiment object loaded from the brain observatory
    # raw: Boolean. False for df/f calcium signal. True for raw fluroescence trace

    # Returns
    # A data frame object containing the trial sorted responses for all spontaeuous activity in the given experiment
    #   session. Each element in the data frame is an array with dims cells X time

    # pr_df: A data frame object containing the trial sorted pupil size for all
    # different natural scenes presented

    # cell_ids: numpy array containing unique cell ids. Index of cell id matches
    # with index of cell X time matrix contained inside dff_df

    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]
    stim_table = exp.get_spontaneous_activity_stimulus_table()
    dff_temp = dict()
    pr_temp = dict()
    dff_temp['spont']= []
    pr_temp['spont']= []

    for i in range(0, len(stim_table['start'].values)):
        start = stim_table['start'][i]
        end = stim_table['end'][i]
        dff_temp['spont'].append(dff[:,start:end])
        pr_temp['spont'].append(pr[start:end])

    columns = sorted(dff_temp.keys())
    dff_df = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    pr_df = pd.DataFrame(data=pr_temp, columns=columns)
    return dff_df, pr_df, cell_ids

def get_ns_specific_fluorescence_traces(exp, raw):

    # Arguments:
    # exp: Individual experiment object loaded from the brain observatory
    # raw: Boolean. False for df/f calcium signal. True for raw fluroescence trace

    # Returns
    # dff_df: A data frame object containing the trial sorted responses for all natural scenes presented in the given experiment
    #   session. Each element in the data frame is an array with dims cells X time

    # pr_df: A data frame object containing the trial sorted pupil size for all
    # different natural scenes presented

    # cell_ids: numpy array containing unique cell ids. Index of cell id matches
    # with index of cell X time matrix contained inside dff_df

    cell_ids = exp.get_cell_specimen_ids()

    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]

    stim = 'natural_scenes'
    stim_table = exp.get_stimulus_table(stim)
    unique_stim = np.sort(exp.get_stimulus_table('natural_scenes')['frame'].unique(), axis=None)
    dff_temp = dict()
    pr_temp = dict()
    for i, u_s in enumerate(unique_stim):
        start = stim_table['start'][stim_table['frame']==u_s].values
        end = start + 7#stim_table['end'][stim_table['frame']==u_s].values
        dff_temp[u_s] = []
        pr_temp[u_s] = []
        for j in range(0, len(start)):
                dff_temp[u_s].append(dff[:,start[j]:end[j]])
                pr_temp[u_s].append(pr[start[j]:end[j]])
    columns = sorted(dff_temp.keys())
    dff_df = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    pr_df = pd.DataFrame(data=pr_temp, columns = columns)
    return dff_df, pr_df, cell_ids
