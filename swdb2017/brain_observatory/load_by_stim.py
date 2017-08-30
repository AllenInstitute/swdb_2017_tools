import numpy as np
import pandas as pd

def get_grating_specific_traces(exp, raw):

    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]
    t, pl = exp.get_pupil_location(as_spherical = False)
    pl_x = pl_x[:,0]
    pl_y = pl_y[:,1]
    stim_table = exp.get_stimulus_table('static_gratings')

    stim_id = []
    for i, s in enumerate(stim_table['orientation']):
        stim_id.append(",".join([str(stim_table['orientation'][i]), str(stim_table['spatial_frequency'][i]),
                                     str(stim_table['phase'][i])]))
    stim_table['id'] = stim_id # add column with unique id

    stim_ids = list(set(stim_id)) # make list of all possible unique ids
    dff_temp = dict()
    columns = stim_ids
    df = pd.DataFrame([], index = range(0,50), columns = columns)
    pr_df = pd.DataFrame([], index = range(0,50), columns = columns)
    t_df = pd.DataFrame([], index = range(0,50), columns = columns)
    for j, us in enumerate(stim_ids):
        start = stim_table['start'][stim_table['id']==us].values
        end = stim_table['end'][stim_table['id']==us].values

        for i in range(0, len(start)):
                df[us][i]=(dff[:,start[i]:end[i]])
                pr_df[us][i] = pr[start[i]:end[i]]
                t_df[us][i] = t[start[i]:end[i]]
    return df, pr_df, cell_ids, t_df

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
    t, pl = exp.get_pupil_location(as_spherical = False)
    pl_x = pl_x[:,0]
    pl_y = pl_y[:,1]
    stim_table = exp.get_spontaneous_activity_stimulus_table()
    dff_temp = dict()
    pr_temp = dict()
    t_temp = dict()
    dff_temp['spont']= []
    pr_temp['spont']= []
    t_temp['spont'] = []
    for i in range(0, len(stim_table['start'].values)):
        start = stim_table['start'][i]
        end = stim_table['end'][i]
        dff_temp['spont'].append(dff[:,start:end])
        pr_temp['spont'].append(pr[start:end])
        t_temp['spont'].append(t[start:end])
    columns = sorted(dff_temp.keys())
    dff_df = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    pr_df = pd.DataFrame(data=pr_temp, columns=columns)
    t_df = pd.DataFrame(data=t_temp, columns = columns)
    return dff_df, pr_df, cell_ids, t_df

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
    t, pl = exp.get_pupil_location(as_spherical = False)
    stim = 'natural_scenes'
    stim_table = exp.get_stimulus_table(stim)
    unique_stim = np.sort(exp.get_stimulus_table('natural_scenes')['frame'].unique(), axis=None)
    dff_temp = dict()
    pr_temp = dict()
    t_temp = dict()
    pl_temp = dict()
    for i, u_s in enumerate(unique_stim):
        start = stim_table['start'][stim_table['frame']==u_s].values
        end = start + 7#stim_table['end'][stim_table['frame']==u_s].values
        dff_temp[u_s] = []
        pr_temp[u_s] = []
        t_temp[u_s] = []
        for j in range(0, len(start)):
                dff_temp[u_s].append(dff[:,start[j]:end[j]])
                pr_temp[u_s].append(pr[start[j]:end[j]])
                t_temp[u_s].append(t[start[i]:end[i]])
    columns = sorted(dff_temp.keys())
    dff_df = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    pr_df = pd.DataFrame(data=pr_temp, columns = columns)
    t_df = pd.DataFrame(data=t_temp, columns = columns)
    return dff_df, pr_df, cell_ids, t_df

def get_ns_dff_by_trial(exp, cell_specimen_ids=None):
    if cell_specimen_ids is None:
        cell_specimen_ids = exp.get_cell_specimen_ids()

    t, dff = exp.get_dff_traces()        # Read in calcium signal

    stim = 'natural_scenes'
    stim_table = exp.get_stimulus_table(stim)
    unique_stim = np.sort(exp.get_stimulus_table('natural_scenes')['frame'].unique(), axis=None)
    dff_temp = dict()
    for i, u_s in enumerate(unique_stim):
        start = stim_table['start'][stim_table['frame']==u_s].values
        end = start + 7#stim_table['end'][stim_table['frame']==u_s].values
        dff_temp[u_s] = []
        for j in range(0, len(start)):
                dff_temp[u_s].append(dff[:,start[j]:end[j]])
    columns = sorted(dff_temp.keys())
    dff_df = pd.DataFrame(data=dff_temp, columns=columns)
    return dff_df
