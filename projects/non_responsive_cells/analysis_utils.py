##################################################################
# layer4_utils.py
#   Contains utility functions to analyze neural spike data
#   psachdeva
##################################################################
import numpy as np
import pandas as pd
from scipy.signal import convolve

def get_isi(filepath = None, spike_times_per_neuron = None):
    '''Returns a dictionary containing the interspike intervals for each neuron in a simulation.
    
    Parameters
    ----------
        Function accepts one of two parameters: 
        
        filepath : string
            the filepath to the .dat file containing the spike outputs for a simulation
        
        spike_times_per_neuron : dict
            keys are neuron ids and values are spike times for each neuron
            
    Returns
    -------
        isi : dict
            keys are neuron ids and values are the interspike intervals for each neuron
    '''
    if (filepath is None) and (spike_times_per_neuron is not None):
        isi = {neuron_id : np.diff(spike_times_per_neuron[neuron_id]) for neuron_id in spike_times_per_neuron.keys()}
    elif (filepath is not None) and (spike_times_per_neuron is None):
        spike_times_per_neuron = get_spike_times_per_neuron(filepath)
        isi = {neuron_id : np.diff(spike_times_per_neuron[neuron_id]) for neuron_id in spike_times_per_neuron.keys()}
    else:
        raise ValueError('Function accepts either a filepath or spike times per neuron dict.')
    return isi

def binarize_spike_times(spike_times, t_start, t_finish, bin_size):
    '''Binarizes an array of spike times.
    
    Parameters
    ----------
        spike_times : numpy array
            contains the times, in seconds, of the spikes for a given neuron
        
        t_start: float
            the starting time to bin over, in seconds
        
        t_finish: float
            the ending time to bin over, in seconds
        
        bin_size : float
            the length, in seconds, of each bin
            
    Returns
    -------
        spike_train : numpy array
            binary array indicating which bins contain spikes
    '''
    # create bins and apply to the spike times
    bins = np.arange(t_start, t_finish, bin_size)
    spike_train = np.histogram(spike_times, bins)[0]
    return spike_train

def binarize_spike_times_population(spike_times_per_neuron, t_start, t_finish, bin_size):
    '''Binarizes a dict containing the spike times for a variety of neurons.
    
    Parameters
    ----------
        spike_times_per_neuron : dict
            key values are neuron ids (ints) and values are numpy arrays containing spike times.
        
        t_start: float
            the starting time to bin over, in seconds
        
        t_finish: float
            the ending time to bin over, in seconds
        
        bin_size : float
            the length, in seconds, of each bin
            
    Returns
    -------
        spike_train : dict
            dict containing neuron ids as keys and binary arrays indicating which bins contain spikes
    '''
    # create bins
    bins = np.arange(t_start, t_finish, bin_size)
    # get the number of neurons
    num_neurons = len(spike_times_per_neuron.keys())
    # set up data storage arrays
    neuron_ids = np.zeros(num_neurons)
    spike_trains = np.zeros((num_neurons, num_bins))
    # iterate over spike times and bin each one
    spike_train = {neuron_id : np.histogram(spike_times_per_neuron[neuron_id], bins)[0] for neuron_id in spike_times_per_neuron}
    return spike_train

def create_gaussian_kernel(bin_size, sigma):
    '''Creates Gaussian kernel that can be convolved with a spike train. 
    
    Parameters
    ----------
        bin_size : float
            the size, in seconds, that each time bin will be
        
        sigma : float
            standard deviation of the Gaussian
    
    Returns
    -------
        kernel : numpy array
            numpy array containing the normalized kernel
    '''
    # time window is (-5sig, 5sig), per Bruno et al. 2017
    ts = np.arange(-5 * sigma, 5 * sigma, bin_size)
    # apply the gaussian across time window with prescribed stdev
    kernel = 1./np.sqrt(2 * np.pi * sigma**2) * np.exp(-ts**2/(2 * sigma**2))
    # normalize
    kernel = kernel/np.sum(kernel)
    return kernel

def convolve_spike_train(spike_times, t_start, t_finish, bin_size, sigma):
    '''Convolves an array of spike times with a Gaussian kernel.
    
    Parameters
    ----------
        spike_times: numpy array
            contains the spike times, in seconds, for a given neuron
            
        t_start: float
            the starting time to bin over, in seconds
        
        t_finish: float
            the ending time to bin over, in seconds
        
        bin_size : float
            the length, in seconds, of each bin  
            
        sigma : float
            standard deviation of the Gaussian kernel to apply
            
    Returns
    -------
        
    '''
    # binarize spike times
    binarized = binarize_spike_times(spike_times, t_start, t_finish, bin_size)
    # create gaussian kernel
    kernel = create_gaussian_kernel(bin_size, sigma)
    # perform the convolution
    convolution = convolve(kernel, binarized)
    # adjust convolution: determine how much excess there is, then strip off both sides
    excess = len(convolution) - len(binarized)
    cleave_index = int(excess/2)
    # adjustment in case excess is odd 
    adjust = bloat % 2
    return convolution[(cleave + adjust):-cleave]