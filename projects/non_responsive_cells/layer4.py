import os
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.decomposition import PCA

class layer4:
	def __init__(directory, model, stimulus, stimulus_id, trial):
		'''Class for a given Layer 4 simulation. Contains functions to
		extract the spike trains and perform analyses on them. 

		Attributes
		----------
		filepath : string
			the filepath to the .dat file containing the spike times for the given simulation

		directory : string
			the base directory containing all layer4 simulations

		model : string
			the Layer 4 model, either 'll1', 'll2', or 'll3'

		stimulus : string
			the type of stimulus used in the simulation

		stimulus_id : int or string
			id identifying specific stimulus within a set

		trial : int
			trial number
		'''
		self.directory = directory
		self.model = model
		self.stimulus = stimulus
		self.stimulus_id = stimulus_id
		self.trial = trial
		self.get_spike_filepath()
		self.get_spike_times_per_neuron()

	def get_spike_filepath():
		'''Determines and stores the filepath to the dat file containing the spike times from a given Layer 4 simulation.
		
		Returns
		-------
		filepath : string
			the filepath to the .dat file containing the spike times for the desired simulation
			IOError if file not found
		'''
		
		### TODO : properly handle nat images, 
		if self.stimulus == 'spont':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus,
									'output_%s_spont_%d_sd278' %(self.model, self.trial),
									'spk.dat')
		elif self.stimulus == 'gratings':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus,
									'output_%s_g%d_%d_sd278' %(self.model, self.stimulus_id, self.trial),
									'spk.dat')
		elif self.stimulus == 'flashes':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus,
									'output_%s_flash_%d_%d_sd278' %(self.model, self.stimulus_id, self.trial),
									'spk.dat')
		elif self.stimulus == 'natural_images':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus, 
									'output_%s_imseq_%d_%d_sd278' %(self.model, self.stimulus_id, self.trial),
									'spk.dat')
		elif self.stimulus == 'natural_movies':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus,
									'output_%s_TouchOfEvil_frames_%s_%d_sd278' %(self.model, self.stimulus_id, self.trial),
									'spk.dat')
		else:
			return ValueError('Stimulus type does not exist.')
		# check whether file exists
		if not os.path.isfile(self.filepath):
			raise IOError('File not found.')

	def get_spikes():
		'''Returns the array containing the spike times for a given simulation specified in a filepath.

		Returns
		-------
		spikes : numpy array
			contains the spike times in one column and neuron ids in the second
		'''
		spikes = np.genfromtxt(self.filepath, delimiter = ' ')
		return spikes

	def get_spike_times_per_neuron():
		'''Returns a dictionary containing the spike times for each neuron id in a simulation.
		
		Returns
		-------
		spike_times_per_neuron : dict
			keys are neuron ids, values are numpy arrays containing the spike times
		'''
		# extract spikes from file
		spikes = get_spikes(self.filepath)
		# create dataframe 
		spikes_df = pd.DataFrame(spikes, columns = ['times', 'neuron_id'])
		# group by neuron
		spikes_df_grouped = spikes_df.groupby('neuron_id')
		# create dictionary containing neuron ids as keys and arrays of spike times as values
		self.spike_times_per_neuron = {}
		# iterate over groups, adding each neuron id and corresponding spike times to the dict
		for neuron_id in spikes_df_grouped.groups.keys():
			self.spike_times_per_neuron[int(neuron_id)] = np.array(spikes_df_grouped.get_group(neuron_id)['times'])

		return self.spike_times_per_neuron

	def spike_times_dict_to_array():
		'''Converts a dictionary with neuron ids as keys and arrays of spike times as values
		to a padded array of spike times, with the neuron ids extracted.
		
		Returns
		-------
		neuron_ids : numpy array
			contains the neuron ids from the dictionary
			
		spike_times : numpy array
			each row corresponds to the spike times for a different neuron. rows are ordered in the same
			way as neuron_ids. lastly, each row is padded with -1s if its corresponding neuron had less
			than the max number of spikes.
		'''
		# get the total number of neurons
		num_neurons = len(spike_times_per_neuron.keys())
		# get the maximum number of spikes a neuron had
		max_num_spikes = len(spike_times_per_neuron[max(spike_times_per_neuron, key = lambda neuron_id : len(spike_times_per_neuron[neuron_id]))])
		# set up neuron id array and spike time array
		neuron_ids = np.zeros(len(spike_times_per_neuron.keys()))
		spike_times = np.zeros((num_neurons, max_num_spikes))
		# iterate over the neuron ids
		for idx, neuron_id in enumerate(spike_times_per_neuron.keys()):
			# extract neuron id, spikes, and number of spikes
			neuron_ids[idx] = int(neuron_id)
			spikes = spike_times_per_neuron[neuron_id]
			num_spikes = len(spikes)
			# amount of padding needed
			padding = max_num_spikes - num_spikes 
			# add spikes along with padding to the spike times array
			spike_times[idx, :] = np.lib.pad(spikes, (0, padding), 'constant', constant_values = (0, -1))
			
		return neuron_ids, spike_times

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

	def get_isi():
		'''Returns a dictionary containing the interspike intervals for each neuron in a simulation.
		
		Returns
		-------
			isi : dict
				keys are neuron ids and values are the interspike intervals for each neuron
		'''
		isi = {neuron_id : np.diff(self.spike_times_per_neuron[neuron_id]) for neuron_id in self.spike_times_per_neuron.keys()}
		return isi

	def binarize_spike_times_population(t_start, t_finish, bin_size):
		'''Binarizes a dict containing the spike times for a variety of neurons.
		
		Parameters
		----------
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
		# create bins
		bins = np.arange(t_start, t_finish, bin_size)
		# get the number of neurons
		num_neurons = len(spike_times_per_neuron.keys())
		# set up data storage arrays
		neuron_ids = np.zeros(num_neurons)
		spike_trains = np.zeros((num_neurons, len(bins) - 1))
		# iterate over spike times and bin each one
		for idx, neuron_id in enumerate(spike_times_per_neuron):
			neuron_ids[idx] = neuron_id
			spike_trains[idx, :] = np.histogram(spike_times_per_neuron[neuron_id], bins)[0]

		return neuron_ids, spike_trains

	def convolve_spike_population(t_start, t_finish, bin_size, percentile):
		'''Binarizes a population of spike times and then convolves the entire population with a 
		Gaussian kernel.
		
		Parameters
		----------				
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
				num_neurons x num_convolutions array containing the convolution of each neuron in each
				row
		'''
		# binarize spike times
		neuron_ids, binarized = binarize_spike_times_population(self.spike_times_per_neuron, t_start, t_finish, bin_size)
		# choose sigma
		isi = self.get_isi()
		sigma = np.percentile(np.concatenate(isi.values()), percentile)/np.sqrt(12)
		# create gaussian kernel
		kernel = create_gaussian_kernel(bin_size, sigma)
		# perform the convolution
		num_neurons, num_bins = binarized.shape
		convolution = np.zeros((num_neurons, num_bins + len(kernel) - 1))
		for idx in range(num_neurons):
			convolution[idx, :] = convolve(kernel, binarized[idx, :])

		return convolution
