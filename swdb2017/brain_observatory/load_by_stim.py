import numpy as np
import pandas as pd

def get_grating_specific_traces(exp, raw):

    '''
    Arguments:
    ---------------------------------------------------------------------------
    exp: Individual experiment object loaded from the brain observatory
    raw: Boolean. False for df/f calcium signal. True for raw fluroescence trace

    Returns:
    ----------------------------------------------------------------------------
    output: Dictionary containing the following fields

        fluorescence: data frame object containing the trial sorted responses for all
        static grating activity in the given experiment session. Each element in the
        data frame is an array with dims cells X time

        pupil size: A data frame object containing the trial sorted pupil size for all
        different natural scenes presented

        pupil location: A data frame object containing the trial sorted pupil location for all
        different natural scenes presented

        running speed: A data frame object containing the trial sorted running speed for all
        different natural scenes presented

        time: A data frame object containing the trial sorted times stamps for all
        different natural scenes presented. Aligns with the previous dictionary fields

    cell_ids: numpy array containing unique cell ids. Index of cell id matches
    with row of cell X time matrix contained inside the fluroescence field of the
    output dictionary
    '''

    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]
    t, pl = exp.get_pupil_location(as_spherical = False)
    t, rs = exp.get_running_speed()
    stim_table = exp.get_stimulus_table('static_gratings')

    stim_id = []
    for i, s in enumerate(stim_table['orientation']):
        stim_id.append(",".join([str(stim_table['orientation'][i]), str(stim_table['spatial_frequency'][i]),
                                     str(stim_table['phase'][i])]))
    stim_table['id'] = stim_id # add column with unique id

    stim_ids = list(set(stim_id)) # make list of all possible unique ids
    columns = stim_ids
    df = pd.DataFrame([], index = range(0,50), columns = columns)
    pr_df = pd.DataFrame([], index = range(0,50), columns = columns)
    t_df = pd.DataFrame([], index = range(0,50), columns = columns)
    pl_df = pd.DataFrame([], index = range(0,50), columns = columns)
    rs_df = pd.DataFrame([], index = range(0,50), columns = columns)
    for j, us in enumerate(stim_ids):
        start = stim_table['start'][stim_table['id']==us].values
        end = stim_table['end'][stim_table['id']==us].values

        for i in range(0, len(start)):
                df[us][i]=(dff[:,start[i]:end[i]])
                pr_df[us][i] = pr[start[i]:end[i]]
                pl_df[us][i] = pl[start[i]:end[i], :]
                t_df[us][i] = t[start[i]:end[i]]
                rs_df[us[i]] = rs[start[i]:end[i]]
    output = dict()
    output['fluroescence']
    output['pupil size']
    output['pupil location']
    output['time']
    output['running speed']
    return output, cell_ids

def get_spont_specific_fluorescence_traces(exp, raw):
    '''
    Arguments:
    ---------------------------------------------------------------------------
    exp: Individual experiment object loaded from the brain observatory
    raw: Boolean. False for df/f calcium signal. True for raw fluroescence trace

    Returns:
    ----------------------------------------------------------------------------
    output: Dictionary containing the following fields

        fluorescence: data frame object containing the trial sorted responses for all
        spontaeuous activity in the given experiment session. Each element in the
        data frame is an array with dims cells X time

        pupil size: A data frame object containing the trial sorted pupil size for all
        different natural scenes presented

        pupil location: A data frame object containing the trial sorted pupil location for all
        different natural scenes presented

        running speed: A data frame object containing the trial sorted running speed for all
        different natural scenes presented

        time: A data frame object containing the trial sorted times stamps for all
        different natural scenes presented. Aligns with the previous dictionary fields

    cell_ids: numpy array containing unique cell ids. Index of cell id matches
    with row of cell X time matrix contained inside the fluroescence field of the
    output dictionary
    '''

    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]
    t, pl = exp.get_pupil_location(as_spherical = False)
    t, rs = exp.get_running_speed()
    stim_table = exp.get_spontaneous_activity_stimulus_table()
    dff_temp = dict()
    pr_temp = dict()
    t_temp = dict()
    pl_temp = dict()
    rs_temp = dict()
    dff_temp['spont']= []
    pr_temp['spont']= []
    pl_temp['spont'] = []
    t_temp['spont'] = []
    rs_temp['spont'] = []

    for i in range(0, len(stim_table['start'].values)):
        start = stim_table['start'][i]
        end = stim_table['end'][i]
        dff_temp['spont'].append(dff[:,start:end])
        pr_temp['spont'].append(pr[start:end])
        pl_temp['spont'].append(pl[start:end,:])
        t_temp['spont'].append(t[start:end])
        rs_temp['spont'].append(rs[start:end])

    output = dict()
    columns = sorted(dff_temp.keys())
    output['fluorescence'] = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    output['pupil size'] = pd.DataFrame(data=pr_temp, columns=columns)
    output['time']= pd.DataFrame(data=t_temp, columns = columns)
    output['pupil location'] = pd.DataFrame(data=pl_temp, columns = columns)
    output['running speed'] = pd.DataFrame(data = rs_temp, columns = columns)

    return output, cell_ids

def get_ns_specific_fluorescence_traces(exp, raw):

    '''
    Arguments:
    ---------------------------------------------------------------------------
    exp: Individual experiment object loaded from the brain observatory
    raw: Boolean. False for df/f calcium signal. True for raw fluroescence trace

    Returns:
    ----------------------------------------------------------------------------
    output: Dictionary containing the following fields

        fluorescence: data frame object containing the trial sorted responses for all
        natural scenes activity in the given experiment session. Each element in the
        data frame is an array with dims cells X time

        pupil size: A data frame object containing the trial sorted pupil size for all
        different natural scenes presented

        pupil location: A data frame object containing the trial sorted pupil location for all
        different natural scenes presented

        running speed: A data frame object containing the trial sorted running speed for all
        different natural scenes presented

        time: A data frame object containing the trial sorted times stamps for all
        different natural scenes presented. Aligns with the previous dictionary fields

    cell_ids: numpy array containing unique cell ids. Index of cell id matches
    with row of cell X time matrix contained inside the fluroescence field of the
    output dictionary
    '''

    cell_ids = exp.get_cell_specimen_ids()

    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()        # Read in calcium signal

    pr = exp.get_pupil_size()
    pr = pr[1]
    t, pl = exp.get_pupil_location(as_spherical = False)
    t, rs = exp.get_running_speed()
    stim = 'natural_scenes'
    stim_table = exp.get_stimulus_table(stim)
    unique_stim = np.sort(exp.get_stimulus_table('natural_scenes')['frame'].unique(), axis=None)
    dff_temp = dict()
    pr_temp = dict()
    t_temp = dict()
    pl_temp = dict()
    rs_temp = dict()
    for i, u_s in enumerate(unique_stim):
        start = stim_table['start'][stim_table['frame']==u_s].values
        end = start + 7#stim_table['end'][stim_table['frame']==u_s].values
        dff_temp[u_s] = []
        pr_temp[u_s] = []
        pl_temp[u_s] = []
        t_temp[u_s] = []
        rs_temp[u_s] = []
        for j in range(0, len(start)):
                dff_temp[u_s].append(dff[:,start[j]:end[j]])
                pr_temp[u_s].append(pr[start[j]:end[j]])
                pl_temp[u_s].append(pl[start[j]:end[j], :])
                t_temp[u_s].append(t[start[j]:end[j]])
                rs_temp[u_s].append(rs[start[j]:end[j]])
    columns = sorted(dff_temp.keys())
    output['fluorescence'] = pd.DataFrame(data=dff_temp, columns=columns)
    columns = sorted(pr_temp.keys())
    output = dict()
    output['pupil size'] = pd.DataFrame(data=pr_temp, columns = columns)
    output['time'] = pd.DataFrame(data=t_temp, columns = columns)
    output['pupil location'] = pd.DataFrame(data=pl_temp, columns = columns)
    output['running speed'] = pd.DataFrame(data= rs_temp, columns = columns)
    return output, cell_ids

def get_ns_dff_by_trial(exp, cell_specimen_ids=None):
    if cell_specimen_ids is None:
        cell_specimen_ids = exp.get_cell_specimen_ids()

    t, dff = exp.get_dff_traces(cell_specimen_ids=cell_specimen_ids)        # Read in calcium signal

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
