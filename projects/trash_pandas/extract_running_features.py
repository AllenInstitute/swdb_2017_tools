# Imports
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter

# Necessary functions
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


def insert_nans(data,idx_nonans):
    '''Reinsert NaNs.

    Parameters
    ----------
    data : array
        data with no NaNs in it
    idx_nonans : array
        boolean containing True where original trace had no NaNs and False where original trace had NaNs

    Returns
    -------
    new_data : array
        data with NaNs reinserted at same indices as original data

    '''

    new_data = np.empty(idx_nonans.shape)
    new_data[:] = np.nan
    new_data[idx_nonans] = data

    return new_data


def extract_running_rate(data, period=6):
    '''Will extract rate of change in input data.

    Parameters
    ----------
    data : array
        contains data

    period : int
        Input for df.diff function. Will take difference between points period apart.

    Returns
    -------
    running_rate : array
        contains running rate values'''

    running_rate = pd.DataFrame(data).diff(periods=period)
    running_rate = running_rate.values.squeeze()

    return running_rate


def extract_smooth_running_rate(dataset, sigma=4):
    '''Extract running rate from smoothed running data

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    smooth_running_rate :
        smoothed pupil area rate with NaNs reinserted
    '''

    running_speed, timestamps = dataset.get_running_speed()

    speed_nonans, idx_nonans = remove_nans(running_speed)

    # Smooth trace
    filt_speed = gaussian_filter(speed_nonans, sigma)
    # Extract rate from smoothed trace
    diff_speed = extract_running_rate(filt_speed, period = 6)

    # Re-insert NaNs to smoothed rate trace
    smooth_running_rate = insert_nans(diff_speed, idx_nonans)

    return smooth_running_rate



