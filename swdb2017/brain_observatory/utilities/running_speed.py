import numpy as np
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
    ax.plot(running_speed, timestamps)
    ax.set_title('expt_session_id: '+str(expt_session_id))
    ax.set_ylabel('run_speed (cm/s)')
    ax.set_xlabel('time (s)')
    return ax
