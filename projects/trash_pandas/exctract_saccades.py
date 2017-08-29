import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def euclidean_distance(x,y):
    ''' Calculate euclidean distance between two points x,y
    
    Parameters
    ----------
    x : array
        x-coordinates for each point
    y : array
        y-coordinates for each point
        
    Returns
    -------
    dist : float
        euclidean distance between x and y'''
        
    dist = np.sqrt((x[0]-x[1])**2 + (y[0]-y[1])**2)
    
    return dist

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

def extract_saccades(data,threshold,sigma = 0):
    ''' Function extract moments where saccades are occuring.
    
    Parameters
    ----------
    data : array 
        input euclidean distance of eye location trace
    
    threshold : int 
        threshold for determining if a saccade is occuring
    
    sigma: int 
        sigma value for guassian zero. Default will not smooth data (sigma = 0).
    
    Returns
    -------
    is_saccade : array
        0s and 1s where value is a 1 if a saccade is occuring. Does not distinguish different directions.'''
    
    filt_data = gaussian_filter(data,sigma)
    diff_data = extract_rate(filt_data,period = 10)
    is_saccade = np.zeros(diff_data.shape)
    is_saccade[abs(diff_data) > threshold] = 1
    
    return is_saccade