# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter

# Behavior Data Functions
def get_filtered_df(boc, targeted_structures=None, stims=None, cre_lines=None):
    """
    Returns pandas dataframe filtered by stimulus inputs and targeted structure inputs from Brain Observatory data.

    Parameters:
    ----------
    boc : BrainObservatoryCache instance
    targeted_structures : list
        brain region acronyms, strings (optional)
    stims : list
        list of stimulus class names, strings (optional)
    cre_lines : list
        list of cre line names, strings (optional)

    Returns:
    -------
    filtered_df : pandas dataframe
        contains experiments only with stim or targeted_structures inputs
    """

    if targeted_structures is None:
        targeted_structures = boc.get_all_targeted_structures()

    if stims is None:
        stims = boc.get_all_stimuli()

    if cre_lines is None:
        cre_lines = boc.get_all_cre_lines()

    filtered_df = pd.DataFrame(
        boc.get_ophys_experiments(stimuli=stims, targeted_structures=targeted_structures, cre_lines=cre_lines, simple=False))

    return filtered_df

def get_running_speed_from_expt_session_id(boc, expt_session_id, remove_outliers=True):
    """
    Author: @marinag
    """
    """
    Get running speed trace for a single experiment session.
    
    Parameters:
    ----------
    boc : BrainObservatoryCache instance
    expt_session_id : int 
        ophys experiment session ID
    remove_outliers : boolean
        if True, replace running trace outlier values (>100, -10) with NaNs
    
    Returns:
    -------
    running_speed : array
        numpy.array of mouse running speed in centimeters per second. Can include NaNs
    timestamps : array
        numpy.array of timestamps corresponding to running speed values
    """

    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=expt_session_id)
    running_speed, timestamps = dataset.get_running_speed()
    if remove_outliers:
        running_speed = remove_running_speed_outliers(running_speed)
    return running_speed, timestamps

def remove_running_speed_outliers(running_speed):
    """
    Replaces outlier points in running speed traces (values >100 & < -20) with NaNs.
    Outlier values likely result from running wheel encoder glitches.

    Parameters:
    ----------
    running_speed : array
        running speed trace

    Returns:
    -------
    run_speed : array
        numpy array running speed trace with outlier values replaced with NaNs
    """

    run_speed = running_speed.copy()
    run_speed[run_speed > 100] = np.nan
    run_speed[run_speed < -20] = np.nan
    return run_speed


def get_behavior_df(boc, df):
    """
    Returns pandas dataframe containing behavior data from Allen Brain Observatory.

    Parameters:
    ---------
    boc : BrainObservatoryCache instance
    df : pandas dataframe
        contains experiment ids for desired behavioral data output

    Returns:
    -------
    behavior_df : pandas dataframe
        contains below
    id : int
        unique experiment session id
    speed_cm_s :  list
        running speed values in cm/s; can include NaNs
    time_stamps : list
        time stamps corresponding to running speed values
    """

    expt_session_ids = df.id.values

    running_speed_list = []
    for expt_id in expt_session_ids[:5]:
        running_speed, timestamps = get_running_speed_from_expt_session_id(boc, expt_id, remove_outliers=True)
        running_speed_list.append([expt_id, running_speed, timestamps])

    behavior_df = pd.DataFrame(running_speed_list, columns=['id', 'speed_cm_s', 'time_stamps'])

    return behavior_df


def plot_running_speed(boc, behavior_df):
    """
    Plot all unprocessed running speed traces for a set of ophys experiments within a
    pandas dataframe containing behavior data

    Parameters
    ----------
    boc : BrainObservatoryCache instance
    behavior_df : pandas dataframe
        contains id (int), speed_cm_s (list), and time_stamps (list)

    Returns
    -------
    ax : axes handle

    """

    x_values = behavior_df['time_stamps']
    y_values = behavior_df['speed_cm_s']
    expt_ids = behavior_df['id']

    for i in range(len(expt_ids)):
        plt.figure()
        plt.plot(x_values[i], y_values[i])
        plt.title('Experiment ID: ' + str(expt_ids[i]))
        plt.xlabel('Time (s)')
        plt.ylabel('Running Speed (cm/s)')

def remove_nans(data):
    ''' Remove NaNs from data.

    Parameters
    ----------
    data : array

    Returns
    -------
    data_nonans : array
        original data with no NaNs
    idx_nonans : array
        boolean containing True where original trace had no NaNs and False where original trace had NaNs'''

    # Keep only indices with non-nan data points
    idx_nonans = ~np.isnan(data)
    temp = data
    data_nonans = temp[idx_nonans]

    return data_nonans, idx_nonans

def smooth_data(df_nonans, keys, sigma):
    '''
    Gaussian smooth data in desired columns specified by keys

    Parameters
    ----------
    df : pandas dataframe
        contains data without Nans
    keys : list
        list of strings with columns containing data to be smoothed
    sigma : int
        sigma value to input to gaussian filter

    Returns
    -------
    df_smooth : pandas dataframe
        original data plus additional columns containing smoothed data
    '''

    new_keys = []
    for i, key in enumerate(keys):
        new_keys.append(key + '_smooth')

    df_smooth = df_nonans.copy()

    # Insert empty columns for smooth data
    for i, key in enumerate(new_keys):
        df_keys = list(df_smooth.keys())
        idx = df_keys.index(keys[i])
        df_smooth.insert(idx + 1, column=key, value="")

    # Smooth data and add to dataframe
    for j in range(len(df_nonans[keys[0]])):
        for i, key in enumerate(new_keys):
            y = gaussian_filter(df_nonans[keys[i]][j], sigma)
            df_smooth[key][j] = y

    return df_smooth


def plot_smoothed_trace(df_smooth, keys):
    '''
    Use a gaussian filter to smooth values from specified key column and plot on top of un-smoothed values.

    Parameters
    ----------
    df_smooth : pandas dataframe
        contains data and time_stamps column without any NaN values.
    keys : list
        with column name (str) from data frame that contains the values that you want to smooth
        (NOT including time_stamps)

    Returns
    -------
    ax: axes handle
        plot of smoothed trace with original trace
    '''

    # Plot figures
    for i, row in df_smooth.iterrows():
        fig, ax = plt.subplots()
        ax.plot(row['time_stamps'], row[keys[0]])
        ax.plot(row['time_stamps'], row[keys[1]])

        # Figure aesthetics
        plt.xlabel('Time (s)')
        plt.ylabel(keys[0])
        plt.title('Experiment ID: ' + str(row['id']))
        plt.legend([keys[0], keys[1]], loc='center left', bbox_to_anchor=(1, 0.5))
        sns.despine()
        sns.set_style('ticks')

# Main behavior data function with plots
def get_processed_behavior_data(boc, targeted_structures=None, stims=None, cre_lines=None, sigma = None):
    '''
    Use a set of filtering criteria to get behavior data and gaussian smoothed traces.

    Parameters
    ----------
    boc : BrainObservatoryCache instance
    targeted_structures : list
        brain region acronyms, strings (optional)
    stims : list
        stimulus class names, strings (optional)
    cre_lines : list
        cre line names, strings (optional)
    sigma : int
        sigma value to use for gaussian smoothing

    Returns
    -------
    df_smooth : pandas dataframe
        both smoothed and un-smoothed data (nan indices dropped)
    ax : axes handle
        plot of smoothed trace with original trace
    '''

    filtered_df = get_filtered_df(boc, targeted_structures=targeted_structures, stims=stims, cre_lines=cre_lines)
    behavior_df = get_behavior_df(boc, filtered_df)
    df_nonans = remove_nans(behavior_df, ['speed_cm_s'])
    df_smooth = smooth_data(df_nonans, ['speed_cm_s'], sigma)
    smoothed_traces = plot_smoothed_trace(df_smooth, ['speed_cm_s', 'speed_cm_s_smooth'])

    return smoothed_traces


def plot_smoothed_running_distributions(df_smooth, key, bins):
    '''
    Plot distribution running speed values that are gaussian filtered.

    Parameters
    ----------
    df_smooth : pandas dataframe
        contains data and time_stamps column without any NaN values.
    key : list
        column names (str) to plot distributions
    bins : int
        number of bins to parse the distribution by

    Returns
    -------
    ax: axes handle
        plot of smoothed trace with original trace
    '''

    for i, row in df_smooth.iterrows():
        fig, ax = plt.subplots()
        ax.hist(row[key[0]], bins=bins)

        plt.xlabel(key[0])
        plt.ylabel('count')
        plt.title('Experiment ID: ' + str(row['id']))
        sns.despine()
        sns.set_style('ticks')


def get_is_running(df_smooth, threshold, figure = True):
    '''
    Assign a binary value to running (1) or not running (0) based on an input threshold

    Parameters
    ----------
    df_smooth : pandas dataframe
        contains data and time_stamps column without any NaN values.
    threshold : int
        speed (cm/s) that you will define running state
    figure : boolean
        if True, return plots of is_running data against time

    Returns
    -------
    df_is_running : pandas dataframe
        copy of original dataframe with an additional column (is_running) with binary values
        denoting running (1) or not running (0)
    ax: axes handle
    '''

    df_is_running = df_smooth.copy()
    is_running = []
    for i, row in df_is_running.iterrows():
        is_running.append([1 if s > threshold else 0 for s in row['speed_cm_s_smooth']])

    df_is_running['is_running'] = is_running

    if figure is True:
        for i, row in df_is_running.iterrows():
            fig, ax = plt.subplots()
            ax.plot(row['time_stamps'], row['is_running'])

    return df_is_running


def plot_fluor_running_traces(boc, exp):
    '''
    Plot calcium fluorescence trace plotted with running speed values that are gaussian filtered.

    Parameters
    ----------
    boc: BrainObservatoryCache instance
    exp : int
        ophys experiment session id

    Returns
    -------
    ax: axes handle
        plot of smoothed trace with original trace
    '''

    # Get traces
    exp_data = boc.get_ophys_experiment_data(exp)
    fluor_ts, fluor_trace = exp_data.get_dff_traces()
    exp_speed, exp_ts = exp_data.get_running_speed()

    # Get the mean subtraction of the fluorescent trace
    mean_fluor = np.mean(fluor_trace[1])
    mean_speed = np.mean(exp_speed)

    # Plot fluorescent and running traces
    for i in fluor_trace:
        fig, ax = plt.subplots()
        ax.plot(fluor_ts, (fluor_trace[i] - mean_fluor))
        ax.plot(exp_ts, (exp_speed))

    # Figure aesthetics
    plt.xlabel('Time (s)')
    plt.ylabel('DF/F')
    plt.ylim(-10, 100)
    sns.despine()


