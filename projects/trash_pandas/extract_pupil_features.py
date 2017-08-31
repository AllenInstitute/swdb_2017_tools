import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def euclidean_distance(x,y):
    dist = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
    return dist

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


    #Keep only indices with non-nan data points
    idx_nonans = ~np.isnan(data)
    temp = np.array(data)
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

def extract_rate(data,period = 10):
    '''Will extract rate of change in input data.

    Parameters
    ----------
    data : array
    contains pupil data

    period : int
    Input for df.diff function. Will take difference between points period apart.

    Returns
    -------
    rate : array
    contains rate'''

    rate = pd.DataFrame(data).diff(periods = period)
    rate = rate.values.squeeze()

    return rate

def convert_pupil_area_to_diameter(data):
    '''Convert pupil size data (which is area of the pupil in pixels squared) to diameter (in pixels)
    Parameters
    ----------
    data : array containing pupil area trace (pixels^2)

    Returns
    -------
    pupil_diameter : array containing pupil diameter trace (pixels) '''

    pupil_diameter = np.sqrt(data/np.pi)
    return pupil_diameter

def extract_smooth_saccade_rate(dataset,sigma = 4):
    '''Extract saccade rate from smoothed saccade data

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    saccade_rate :
        smoothed saccade rate with NaNs reinserted '''

    timestamps, pupil_loc = dataset.get_pupil_location(as_spherical = False)
    pupil_x = pupil_loc[:,0]
    pupil_y = pupil_loc[:,1]

    dist = []
    for n in range(len(timestamps)):
        dist.append(euclidean_distance([pupil_x[n], 0], [pupil_y[n], 0]))

    dist_nonans, idx_nonans = remove_nans(dist)

    #Smooth trace
    filt_data = gaussian_filter(dist_nonans,sigma)
    #Extract rate from smoothed trace
    diff_data = extract_rate(filt_data,period = 10)
    #Re-insert NaNs to smoothed rate trace
    saccade_rate = insert_nans(diff_data,idx_nonans)

    return saccade_rate

def extract_smooth_pupil_rate(dataset,sigma = 4):
    '''Extract pupil size rate from smoothed pupil data

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    pupil_area_rate :
        smoothed pupil area rate with NaNs reinserted
    pupil_diameter_rate :
        smoothed pupil diameter rate with NaNs reinserted '''

    timestamps, pupil_area = dataset.get_pupil_size()

    pupil_diameter = convert_pupil_area_to_diameter(pupil_area)

    area_nonans, idx_nonans = remove_nans(pupil_area)
    diameter_nonans, idx_nonans = remove_nans(pupil_diameter)

    #Smooth trace
    filt_area = gaussian_filter(area_nonans,sigma)
    filt_diameter = gaussian_filter(diameter_nonans,sigma)
    #Extract rate from smoothed trace
    diff_area = extract_rate(filt_area,period = 10)
    diff_diameter = extract_rate(filt_diameter,period = 10)
    #Re-insert NaNs to smoothed rate trace


    pupil_area_rate = insert_nans(diff_area,idx_nonans)
    pupil_diameter_rate = insert_nans(diff_diameter,idx_nonans)

    return pupil_area_rate, pupil_diameter_rate

def extract_smooth_pupil(dataset,sigma = 4):
    '''Smooth pupil size trace.

    Parameters
    ----------
    dataset : NWB
    sigma : int
        smoothing parameter for gaussian filter function

    Returns
    -------
    pupil_area_smooth :
        smoothed pupil area with NaNs reinserted
    pupil_diameter_smooth :
        smoothed pupil diameter with NaNs reinserted '''

    timestamps, pupil_area = dataset.get_pupil_size()

    pupil_diameter = convert_pupil_area_to_diameter(pupil_area)

    area_nonans, idx_nonans = remove_nans(pupil_area)
    diameter_nonans, idx_nonans = remove_nans(pupil_diameter)

    #Smooth trace
    filt_area = gaussian_filter(area_nonans,sigma)
    filt_diameter = gaussian_filter(diameter_nonans,sigma)

    #Re-insert NaNs to smoothed rate trace
    pupil_area_smooth = insert_nans(filt_area,idx_nonans)
    pupil_diameter_smooth = insert_nans(filt_diameter,idx_nonans)

    return pupil_area_smooth, pupil_diameter_smooth

def is_saccade(saccade_rate,threshold):
    ''' Binary categorization of pupil location trace. 1s where a saccade is occuring, 0s where eye is stationary.

    Parameters
    ----------
    saccade_rate : array
        Smoothed saccade rate
    threshold : int
        Threshold for determining if a saccade is occuring

    Returns
    -------
    is_saccade : array
        Binary array (may contain NaNs)

    '''

    #saccade_nonans, idx_nonans = remove_nans(saccade_rate)
    is_saccade = np.zeros(saccade_rate.shape)
    is_saccade[abs(saccade_rate) > threshold] = 1
    is_saccade[list(np.where(np.isnan(saccade_rate)))] = np.nan

    return is_saccade
