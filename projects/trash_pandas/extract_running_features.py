"""
@author: fionag
"""

# Imports
import os
import numpy as np
from scipy.ndimage.filters import gaussian_filter

# Retrieve brain observatory cache
manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)

    
# Necessary functions to define
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

# USE ME!!
def extract_smooth_running_rate(dataset, sigma=4):
    '''Extract running rate from smoothed running data

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    running_rate :
        smoothed pupil area rate with NaNs reinserted
    pupil_diameter_rate :
        smoothed pupil diameter rate with NaNs reinserted '''

    timestamps, running_speed = dataset.get_running_speed()

    speed_nonans, idx_nonans = remove_nans(running_speed)

    # Smooth trace
    filt_speed = gaussian_filter(speed_nonans, sigma)
    # Extract rate from smoothed trace
    diff_speed = extract_rate(filt_speed, period=10)

    # Re-insert NaNs to smoothed rate trace
    speed_rate = insert_nans(diff_speed, idx_nonans)

    return speed_rate
