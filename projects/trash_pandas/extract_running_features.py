"""
@author: fionag
"""

# Imports
import os
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


def insert_nans(data, idx_nonans):
    '''Reinsert NaNs.

    Parameters
    ----------
    data : np.ndarray
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


def extract_rate(data, period=6):
    '''Will extract rate of change in input data.

    Parameters
    ----------
    data : array
        contains data

    period : int
        Input for df.diff function. Will take difference between points period apart.

    Returns
    -------
    rate : array
        contains rate'''

    rate = pd.DataFrame(data).diff(periods=period)
    rate = rate.values.squeeze()

    return rate


def extract_smooth_running_rate(dataset, sigma=2):
    '''Extract running rate from smoothed running data

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    running_rate :
        smoothed running rate with NaNs reinserted'''

    running_speed, timestamps = dataset.get_running_speed()

    speed_nonans, idx_nonans = remove_nans(running_speed)

    # Smooth trace
    filt_speed = gaussian_filter(speed_nonans, sigma)

    # Extract rate from smoothed trace
    diff_speed = extract_rate(filt_speed, period=10)

    # Re-insert NaNs to smoothed rate trace
    speed_rate = insert_nans(diff_speed, idx_nonans)

    return speed_rate

def extract_smooth_running_speed(dataset, sigma = 2):
    '''Extract smoothed running speed trace

     Parameters
     ----------
     dataset : NWB
     sigma : int
         smoothing parameter for gaussian filter function

     Returns
     -------
     running_speed_smooth :
         smoothed running rate with NaNs reinserted'''

    running_speed, timestamps = dataset.get_running_speed()

    speed_nonans, idx_nonans = remove_nans(running_speed)

    # Smooth trace
    filt_speed = gaussian_filter(speed_nonans, sigma)

    # Re-insert NaNs to smoothed rate trace
    speed_trace = insert_nans(filt_speed, idx_nonans)

    return speed_trace


def extract_is_running(dataset, threshold):
    '''Assign a binary value to running (1) or not running (0) based on an input threshold
    Parameters
    ----------
    dataset : NWB
    threshold : int
        speed (cm/s) that will define running state

    Returns
    -------
    is_running_full : array
        binary values denoting running (1) or not running (0)'''

    running_speed, timestamps = dataset.get_running_speed()

    speed_nonans, idx_nonans = remove_nans(running_speed)

    # Smooth running trace
    filt_speed = gaussian_filter(speed_nonans, sigma=2)

    # Create dataframe for running speed and time stamps
    df_is_running = pd.DataFrame(columns=['filt_speed'])
    df_is_running['filt_speed'] = filt_speed

    # Iterate through dataframe to assign binary values
    is_running = [1 if row['filt_speed'] > threshold else 0 for i, row in df_is_running.iterrows()]

    # Convert dataframe column into an array
    is_running_arr = np.array(is_running)

    # Re-insert Nans
    is_running_full = insert_nans(is_running_arr, idx_nonans)

    return is_running_full
