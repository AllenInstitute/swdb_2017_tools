import os
import numpy as np
import pandas as pd
from scipy.signal import convolve

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

			stimulus_id : int
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

	def get_spike_filepath():
		'''Returns the filepath to the dat file containing the spike times from a given Layer 4 simulation.
		
		Returns
		-------
			filepath : string
				the filepath to the .dat file containing the spike times for the desired simulation
				IOError if file not found
		'''
		
		### TODO : handle non-natural image case
		if self.stimulus == 'gratings':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus,
									'output_%s_g%d_%d_sd278' % (self.model, self.stimulus_id, self.trial),
									'spk.dat')
		elif self.stimulus == 'natural_images':
			self.filepath = os.path.join(self.directory, 'simulations_' + self.model, self.stimulus, 
									'output_%s_imseq_%d_%d_sd278' % (self.model, self.stimulus_id, self.trial),
									'spk.dat')
		
		# check whether file exists
		if os.path.isfile(self.filepath):
			return self.filepath
		else:
			raise IOError('File not found.')

	def get_spikes(filepath):
		'''Returns the array containing the spike times for a given simulation specified in a filepath.
		
		Parameters
		----------
			filepath : string
				the filepath to the .dat file containing the spike times
		
		Returns
		-------
			spikes : numpy array
				contains the spike times in one column and neuron ids in the second
		'''
		spikes = np.genfromtxt(filepath, delimiter = ' ')
		return spikes

	def get_spike_times_per_neuron(filepath):
		'''Returns a dictionary containing the spike times for each neuron id in a simulation.
		
		Parameters
		----------
			filepath : string
				the filepath to the .dat file containing the spike times
		
		Returns
		-------
			spike_times_per_neuron : dict
				keys are neuron ids, values are numpy arrays containing the spike times
		'''
		# extract spikes from file
		spikes = get_spikes(filepath)
		# create dataframe 
		spikes_df = pd.DataFrame(spikes, columns = ['times', 'neuron_id'])
		# group by neuron
		spikes_df_grouped = spikes_df.groupby('neuron_id')
		# create dictionary containing neuron ids as keys and arrays of spike times as values
		spike_times_per_neuron = {}
		# iterate over groups, adding each neuron id and corresponding spike times to the dict
		for neuron_id in spikes_df_grouped.groups.keys():
			spike_times_per_neuron[int(neuron_id)] = np.array(spikes_df_grouped.get_group(neuron_id)['times'])
		return spike_times_per_neuron

	def spike_times_dict_to_array(spike_times_per_neuron):
		'''Converts a dictionary with neuron ids as keys and arrays of spike times as values
		to a padded array of spike times, with the neuron ids extracted.
		
		Parameters
		----------
			spike_times_per_neuron : dict
				key values are neuron ids (ints) and values are numpy arrays containing spike times. importantly,
				each array of spike times is not necessarily the same length.
		
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
