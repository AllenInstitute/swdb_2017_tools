import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


def get_running_speed_from_expt_session_id(boc, expt_session_id, remove_outliers=True):
    """Get running speed trace for a single experiment session.

    Parameters
    ----------
    boc: Brain Observatory Cache instance
    expt_session_id : ophys experiment session ID
    remove_outliers : Boolean. If True, replace running trace outlier values (>100, -10) with NaNs

    Returns
    -------
    running_speed : values of mouse running speed in centimeters per second. Can include NaNs
    timestamps : timestamps corresponding to running speed values 
    """
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=expt_session_id)
    running_speed, timestamps = dataset.get_running_speed()
    if remove_outliers:
        running_speed = remove_running_speed_outliers(running_speed)
    return running_speed, timestamps


def remove_running_speed_outliers(running_speed):
    """Replaces outlier points in running speed traces (values >100 & < -20) with NaNs.
        Outlier values likely result from running wheel encoder glitches.

    Parameters
    ----------
    running_speed : running speed trace 

    Returns
    -------
    run_speed : running speed trace with outlier values replaced with NaNs
    """
    run_speed = running_speed.copy()
    run_speed[run_speed > 100] = np.nan
    run_speed[run_speed < -20] = np.nan
    return run_speed


def plot_running_speed(boc, expt_session_id, ax=None):
    """Plot running speed trace for a specific ophys experiment session

    Parameters
    ----------
    boc: Brain Observatory Cache instance
    expt_session_id : ophys experiment session ID
    ax: axes handle. If None, a figure will be created

    Returns
    -------
    ax : axes handle
    """
    running_speed, timestamps = get_running_speed_from_expt_session_id(boc, expt_session_id)
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 3))
    ax.plot(timestamps, running_speed)
    ax.set_title('expt_session_id: '+str(expt_session_id))
    ax.set_ylabel('run_speed (cm/s)')
    ax.set_xlabel('time (s)')
    return ax


def get_running_speed_list(expt_session_ids):
    """get lists of running speeds and timestamps for selected experiment session IDs

    Parameters
    ----------
    expt_session_ids : list of experiment session IDs

    Returns
    -------
    running_speed_list: list of running_speeds for selected experiment session IDs, with outliers filtered
    timestamps_list : list of timestamps corresponding to running speeds for selected experiment session IDs
    """
    running_speed_list = []
    timestamps_list = []
    for expt_session_id in expt_session_ids:
        running_speed, timestamps = get_running_speed_from_expt_session_id(boc, expt_session_id)
        running_speed_list.append(running_speed)
        timestamps_list.append(timestamps)
    return running_speed_list, timestamps_list


def get_run_df(expt_session_ids):
    """create pandas dataframe with running speed & timestamps plus descriptive statistics
        for running speed traces, for selected experiment session_ids

    Parameters
    ----------
    expt_session_ids : list of experiment session IDs

    Returns
    -------
    run_df : dataframe, each row corresponds to a single experiment session, columns for computed statistics
    """
    running_speed_list, timestamps_list = get_running_speed_list(expt_session_ids)
    run_df_list = []
    for i, expt_session_id in enumerate(expt_session_ids):
        stats_names = ['n_obs', 'min', 'max', 'mean', 'variance', 'skewness', 'kurtosis', 'coeff_variation']
        running_speed = running_speed_list[i]
        timestamps = timestamps_list[i]
        timestamps = timestamps[np.isnan(running_speed) == False]
        running_speed = running_speed[np.isnan(running_speed) == False]

        var = stats.variation(running_speed)
        s = stats.describe(running_speed)
        stats_values = [s[0], s[1][0], s[1][1], s[2], s[3], s[4], s[5], var]

        run_df_list.append([expt_session_id, running_speed, timestamps]+stats_values)
    columns = ['session_id', 'running_speed', 'timestamps']+stats_names
    run_df = pd.DataFrame(run_df_list, columns=columns)
    return run_df


def get_running_speed_diff(running_speed, periods=6):
    """computes difference value between each timepoint in running_speed and the timepoint #periods later

    Parameters
    ----------
    running_speed : running speed timeseries trace
    periods : number of timepoints over which to take the difference

    Returns
    -------
    diff : running speed diff timeseries
    """
    diff = pd.DataFrame(running_speed).diff(periods=periods)
    diff = diff.values.squeeze()
    return diff


def get_running_speed_changes(run_df, period = 6):
    """compute difference of run speed trace in window defined by period,
        define acceleration, decelleration & stationary periods based on value of difference,
        add all to run dataframe

    Parameters
    ----------
    run_df : dataframe with running_speed trace and associated statistics for experiment sessions (rows)
    periods : number of timepoints over which to take the difference

    Returns
    -------
    run_df : dataframe with running_speed trace and associated statistics plus difference masks
        for experiment sessions (rows)
    """
    run_changes_list = []
    for i in range(len(run_df)):
        running_speed = run_df.running_speed.values[i]
        diff = get_running_speed_diff(running_speed, periods=period)
        accelerating_mask = [True if x >= 2 else False for x in diff]
        deccelerating_mask = [True if x <= -2 else False for x in diff]
        stationary_mask = [True if x < 2 else False for x in running_speed]
        run_changes_list.append([run_df.session_id.values[i], diff, accelerating_mask, deccelerating_mask, stationary_mask])
    columns = ['session_id', 'diff', 'accelerating_mask', 'deccelerating_mask', 'stationary_mask']
    run_changes_df = pd.DataFrame(run_changes_list, columns=columns)
    run_df = run_df.merge(run_changes_df, how='inner', on='session_id')
    return run_df


def plot_run_speed_diff(running_speed, xlim=None):
    """plot running speed timeseries

    Parameters
    ----------
    running_speed : running speed trace

    Returns
    -------
    run_speed : running speed trace with outlier values replaced with NaNs
    """
    diff = get_running_speed_diff(running_speed, periods=6)
    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(running_speed, color='b', label='run_speed')
    ax.plot(diff, color='m', label='run_speed_diff')
    ax.set_xlabel('acquisition frames')
    ax.set_ylabel('running_speed (cm/s)')
    ax.set_xlim(xlim)
    ax.set_ylim(-20, 50)
    ax.legend(loc=9)


def plot_running_speed_changes(run_df, session_id, xlim=None, ax=None):
    """Replaces outlier points in running speed traces (values >100 & < -20) with NaNs.
        Outlier values likely result from running wheel encoder glitches.

    Parameters
    ----------
    run_df : dataframe with running_speed trace and associated statistics for experiment sessions (rows)
    session_id : experiment session ID to plot trace for
    xlim : limits of x axis for which to plot (in seconds of imaging session)
    ax : axes handle for plot. If None, a figure instance will be created

    Returns
    -------
    ax : axes handle for plot
    """
    tmp = run_df[run_df.session_id == session_id]
    running_speed = tmp.running_speed.values[0]
    timestamps = tmp.timestamps.values[0]
    # get traces for different running states
    accelerating = running_speed.copy()
    decelerating = running_speed.copy()
    stationary = running_speed.copy()
    # set values outside mask to nan
    accelerating[np.logical_not(tmp.accelerating_mask.values[0])] = np.nan
    decelerating[np.logical_not(tmp.decelerating_mask[0])] = np.nan
    stationary[np.logical_not(tmp.stationary_mask[0])] = np.nan
    if ax is None:
        fig,ax = plt.subplots(figsize=(15,4))
    ax.plot(timestamps, accelerating, color='g', label='accelerating')
    ax.plot(timestamps, decelerating, color='b', label='decelerating')
    ax.plot(timestamps, stationary, color='r', label='stationary')
    ax.set_title('session_id: '+str(session_id))
    ax.set_xlabel('time (seconds)')
    ax.set_ylabel('running_speed (cm/s)')
    if xlim:
        ax.set_xlim(xlim)
    ax.legend(loc=9)
    return ax





