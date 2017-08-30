##################################################################
# layer4_utils.py
#   Contains utility functions to analyze neural spike data
#   psachdeva
##################################################################
import numpy as np
import pandas as pd
from scipy.signal import convolve
from sklearn.decomposition import PCA

def get_isi(spike_times_per_neuron):
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
	isi = {neuron_id : np.diff(spike_times_per_neuron[neuron_id]) for neuron_id in spike_times_per_neuron.keys()}
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

def binarize_spike_times_population(spike_times_per_neuron, t_start, t_finish, bin_size, num_neurons = 45000):
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
		neuron_ids : array
			contains the neuron ids 

		spike_trains : array
			num_neurons x num_bins binary array indicating whether a spike occurred in a bin for a given neuron
	'''
	### TODO: trickiness with number of neurons variable, since some layer 4 neurons don't spike

	# create bins
	bins = np.arange(t_start, t_finish, bin_size)
	# set up data storage arrays
	neuron_ids = np.zeros(len(spike_times_per_neuron.keys()))
	spike_trains = np.zeros((num_neurons, len(bins) - 1))
	# iterate over spike times and bin each one
	for idx, neuron_id in enumerate(spike_times_per_neuron):
		neuron_ids[idx] = neuron_id
		spike_trains[neuron_id, :] = np.histogram(spike_times_per_neuron[neuron_id], bins)[0]

	return neuron_ids, spike_trains

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
	convolution : numpy array
		contains the resulting convolution between the kernel and binarized spike array
	'''
	# binarize spike times
	binarized = binarize_spike_times(spike_times, t_start, t_finish, bin_size)
	# create gaussian kernel
	kernel = create_gaussian_kernel(bin_size, sigma)
	# perform the convolution
	convolution = convolve(kernel, binarized)
	return convolution

def convolve_spike_population(spike_times_per_neuron, t_start, t_finish, bin_size, percentile):
	'''Binarizes a population of spike times and then convolves the entire population with a 
	Gaussian kernel.
	
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
			
	percentile : float
		to obtain the standard deviation of the Gaussian kernel, a percentile of the population
		ISI is used. 
			
	Returns
	-------
	convolution : numpy array
		num_neurons x num_convolutions array containing the convolution of each neuron in each row
	'''
	# binarize spike times
	neuron_ids, binarized = binarize_spike_times_population(spike_times_per_neuron, t_start, t_finish, bin_size)
	# choose sigma
	isi = get_isi(spike_times_per_neuron)
	sigma = np.percentile(np.concatenate(isi.values()), percentile)/np.sqrt(12)
	# create gaussian kernel
	kernel = create_gaussian_kernel(bin_size, sigma)
	# perform the convolution
	num_neurons, num_bins = binarized.shape
	convolution = np.zeros((num_neurons, num_bins + len(kernel) - 1))
	for neuron_id in neuron_ids:
		convolution[neuron_id, :] = convolve(kernel, binarized[neuron_id, :])

	return convolution

def pca_trajectory(spike_times_per_neuron, desired_exp_var = 0.8, num_components = 10, 
				   t_start = 0, t_finish = 3, bin_size = 0.01, percentile = 50):
	'''Binarizes a population of spike times and then convolves the entire population with a 
	Gaussian kernel.
	
	Parameters
	----------
	spike_times_per_neuron : dict
		key values are neuron ids (ints) and values are numpy arrays containing spike times.

	desired_exp_var : float
		the desired fraction of variance to explain (defaulted to 80%)

	num_components: int
		the number of principal components to extract; will increase if they don't explain enough variance
			
	t_start: float
		the starting time to bin over, in seconds
		
	t_finish: float
		the ending time to bin over, in seconds
		
	bin_size : float
		the length, in seconds, of each bin  
			
	percentile : float
		to obtain the standard deviation of the Gaussian kernel, a percentile of the population
		ISI is used. 
			
	Returns
	-------
	pca : PCA object
		the sklearn pca object that was fit to the convolved spike trains

	projections : numpy array
		the projections spike train dataset into the PCA space
	'''
	# convolve the spike trains
	convolution = self.convolve_spike_population(spike_times_per_neuron, t_start, t_finish, bin_size, percentile)
	# define PCA object, starting with 10 components by default (unless otherwise specified)
	pca = PCA(n_components = 10)
	# compute the percentage of explained variance
	exp_var = np.sum(pca.explained_variance_ratio_)
	# if we haven't explained the desired amount of variance, keep bumping up components
	# until we do explain enough or we get more than 15 components
	while (exp_var < desired_exp_var) and (n_components < 15):
		n_components += 1
		pca = PCA(n_components = n_components)
		projections = pca.fit_transform(convolution.T)
		exp_var = np.sum(pca.explained_variance_ratio_)

	return pca, projections

def participation(pca):
	'''Calculates the L1-norm participation, specified by Bruno et al. 2017.

	Parameters
	----------
	pca : PCA  object
		the sklearn pca object that was fit to the convolved spike trains

	Returns
	-------
	rho : numpy array
		the normalized participation of each neuron in the projection
	'''
	# extract the principal components
	W = pca.components_
	# extract eigenvalues
	lambduh = pca.explained_variance_.reshape((pca.n_components_, 1))
	# calculate the participation per Bruno et al. 2017
	rho = np.sum(np.abs(lambduh * W), axis = 0)
	# normalize rho 
	rho = rho/np.sum(rho)
	return rho