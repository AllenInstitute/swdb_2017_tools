# Imports
import numpy as np
import pandas as pd
import swdb2017.brain_observatory.behavior.extract_pupil_features as epf
import swdb2017.brain_observatory.behavior.extract_running_features as erf
from swdb2017.brain_observatory.behavior.correlation_matrix import pearson_corr_coeff

def corr_spont_behavior(exp, raw=False):
    '''
    Calculate correlation between dff (raw = False) or fluorescence (raw = True) traces from spontaneous activity trials
    and behavior features for each cell in specified experiment sessions.
    Default is raw = False

    Parameters
    ----------
    exp : NWB dataset
    raw : boolean
        True will calculate using raw fluorescence traces, False will calculate using dff traces

    Returns
    -------
    spont_behavior_corr_df : pandas dataframe
      contains pearson corelation coefficients split by behavior features being correlated (column) by cell id (row)

    '''

    stim_table = exp.get_spontaneous_activity_stimulus_table()

    # Extract fluorescent traces
    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()  # Read in calcium signal

    # Extract behavior features
    pupil_area_rate, _ = epf.extract_smooth_pupil_rate(exp)
    saccade_rate = epf.extract_smooth_saccade_rate(exp)
    pupil_area_smooth, _ = epf.extract_smooth_pupil(exp)  # Sigma is defaulted to 4
    running_speed_smooth = erf.extract_smooth_running_speed(exp)  # Sigma is deafult = 2
    running_rate_smooth = erf.extract_smooth_running_rate(exp)  # Sigma is default = 2

    # Pre-allocate
    dff_temp = []
    timestamps_temp = []
    pupil_area_rate_temp = []
    saccade_rate_temp = []
    pupil_area_smooth_temp = []
    running_speed_smooth_temp = []
    running_rate_smooth_temp = []

    # Extract traces and behavior for appropriate timestamps for spontaneous
    for i in range(0, len(stim_table['start'].values)):
        start = stim_table['start'][i]
        end = stim_table['end'][i]
        dff_temp.append(dff[:, start:end])
        timestamps_temp.append(t[start:end])
        pupil_area_rate_temp.append(pupil_area_rate[start:end])
        saccade_rate_temp.append(saccade_rate[start:end])
        pupil_area_smooth_temp.append(pupil_area_smooth[start:end])
        running_speed_smooth_temp.append(running_speed_smooth[start:end])
        running_rate_smooth_temp.append(running_rate_smooth[start:end])

    dff_stack = np.hstack(dff_temp)
    timestamps_stack = np.hstack(timestamps_temp)
    pupil_area_rate_stack = np.hstack(pupil_area_rate_temp)
    saccade_rate_stack = np.hstack(saccade_rate_temp)
    pupil_area_smooth_stack = np.hstack(pupil_area_smooth_temp)
    running_speed_smooth_stack = np.hstack(running_speed_smooth_temp)
    running_rate_smooth_stack = np.hstack(running_rate_smooth_temp)

    # Calculate correlations
    nCell = len(cell_ids)
    spont_behavior_corr = np.zeros((nCell, 5))

    for cellie in range(nCell):
        spont_behavior_corr[cellie, 0] = pearson_corr_coeff(dff_stack[cellie, :], pupil_area_rate_stack)
        spont_behavior_corr[cellie, 1] = pearson_corr_coeff(dff_stack[cellie, :], saccade_rate_stack)
        spont_behavior_corr[cellie, 2] = pearson_corr_coeff(dff_stack[cellie, :], pupil_area_smooth_stack)
        spont_behavior_corr[cellie, 3] = pearson_corr_coeff(dff_stack[cellie, :], running_speed_smooth_stack)
        spont_behavior_corr[cellie, 4] = pearson_corr_coeff(dff_stack[cellie, :], running_rate_smooth_stack)

    # Create dataframe to store data
    feature_ids = ['pupil_area_rate', 'saccade_rate', 'pupil_area_smooth', 'running_speed_smooth',
                   'running_rate_smooth']

    spont_behavior_corr_df = pd.DataFrame(columns=['cell_ids'])
    spont_behavior_corr_df['cell_ids'] = cell_ids

    for i, idx in enumerate(feature_ids):
        spont_behavior_corr_df[idx] = spont_behavior_corr[:, i]

    return spont_behavior_corr_df


def corr_ns_behavior(exp, raw=False):
    '''
    Calculate correlation between dff (raw = False) or fluorescence (raw = True) traces from natural scences stimulus trials
    and behavior features for each cell in specified experiment sessions.
    Default is raw = False

    Parameters
    ----------
    exp : NWB dataset
    raw : boolean
        True will calculate using raw fluorescence traces, False will calculate using dff traces

    Returns
    -------
    ns_behavior_corr_df : pandas dataframe
      contains pearson correlation coefficients split by behavior features being correlated (column) by cell id (row)
    '''

    stim = 'natural_scenes'
    stim_table = exp.get_stimulus_table(stim)

    # Extract fluoresences traces
    cell_ids = exp.get_cell_specimen_ids()
    if raw:
        t, dff = exp.get_fluorescence_traces()  # Read in calcium signal
    else:
        t, dff = exp.get_dff_traces()  # Read in calcium signal

    # Extract behavior features
    pupil_area_rate, _ = epf.extract_smooth_pupil_rate(exp)
    saccade_rate = epf.extract_smooth_saccade_rate(exp)
    pupil_area_smooth, _ = epf.extract_smooth_pupil(exp)  # Sigma is defaulted to 4
    running_speed_smooth = erf.extract_smooth_running_speed(exp)  # Sigma is deafult = 2
    running_rate_smooth = erf.extract_smooth_running_rate(exp)  # Sigma is default = 2

    # Pre-allocate
    start = stim_table['start'].values
    end = stim_table['end'].values
    dff_temp = []
    timestamps_temp = []
    pupil_area_rate_temp = []
    saccade_rate_temp = []
    pupil_area_smooth_temp = []
    running_speed_smooth_temp = []
    running_rate_smooth_temp = []

    # Concatenate traces for all natural scenes
    for j in range(0, len(start)):
        dff_temp.append(dff[:, start[j]:end[j]])
        timestamps_temp.append(t[start[j]:end[j]])
        pupil_area_rate_temp.append(pupil_area_rate[start[j]:end[j]])
        saccade_rate_temp.append(saccade_rate[start[j]:end[j]])
        pupil_area_smooth_temp.append(pupil_area_smooth[start[j]:end[j]])
        running_speed_smooth_temp.append(running_speed_smooth[start[j]:end[j]])
        running_rate_smooth_temp.append(running_rate_smooth[start[j]:end[j]])

    dff_stack = np.hstack(dff_temp)
    timestamps_stack = np.hstack(timestamps_temp)
    pupil_area_rate_stack = np.hstack(pupil_area_rate_temp)
    saccade_rate_stack = np.hstack(saccade_rate_temp)
    pupil_area_smooth_stack = np.hstack(pupil_area_smooth_temp)
    running_speed_smooth_stack = np.hstack(running_speed_smooth_temp)
    running_rate_smooth_stack = np.hstack(running_rate_smooth_temp)

    # Calculate correlations
    nCell = len(cell_ids)
    ns_behavior_corr = np.zeros((nCell, 5))

    for cellie in range(nCell):
        ns_behavior_corr[cellie, 0] = pearson_corr_coeff(dff_stack[cellie, :], pupil_area_rate_stack)
        ns_behavior_corr[cellie, 1] = pearson_corr_coeff(dff_stack[cellie, :], saccade_rate_stack)
        ns_behavior_corr[cellie, 2] = pearson_corr_coeff(dff_stack[cellie, :], pupil_area_smooth_stack)
        ns_behavior_corr[cellie, 3] = pearson_corr_coeff(dff_stack[cellie, :], running_speed_smooth_stack)
        ns_behavior_corr[cellie, 4] = pearson_corr_coeff(dff_stack[cellie, :], running_rate_smooth_stack)

    # Create dataframe to store data
    feature_ids = ['pupil_area_rate', 'saccade_rate', 'pupil_area_smooth', 'running_speed_smooth',
                   'running_rate_smooth']

    ns_behavior_corr_df = pd.DataFrame(columns=['cell_ids'])
    ns_behavior_corr_df['cell_ids'] = cell_ids

    for i, idx in enumerate(feature_ids):
        ns_behavior_corr_df[idx] = ns_behavior_corr[:, i]

    return ns_behavior_corr_df


def count_corr_cells(df, threshold):
    """
    Counts number of values in each column (behavior feature correlation) above
    an input threshold and below -threshold separately

    Parameters:
    ----------
    df : pandas dataframe
        contains pearson correlation coefficients from behavior features
    threshold : float
        between 0 and 1

    Returns:
    -------
    pos_count : array
        contains a summed value of significant correlations for each feature
    neg_count : array
        contains a summed value of significant correaltion for each feature
    """

    # Initialize components
    df_keys = df.columns[1:]
    is_sig_pos = np.zeros((len(df[df.columns[0]].values), len(df_keys)))
    is_sig_neg = np.zeros((len(df[df.columns[0]].values), len(df_keys)))

    # Assign values based on threshold
    for i, key in enumerate(df_keys):
        vals = df[key].values
        is_sig_pos[(vals > threshold), i] = 1
        is_sig_neg[(vals < -threshold), i] = 1

    # Sum values
    pos_count = np.sum(is_sig_pos, axis=0)
    neg_count = np.sum(is_sig_neg, axis=0)

    return pos_count, neg_count