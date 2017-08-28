##################################################################
# layer4_utils.py
# 	Contains utility functions for use with layer 4 simulation data
# 	psachdeva
##################################################################
import os
import numpy as np
import pandas as pd


### Functions to extract file information 

def extract_stimuli(drive_path, model):
	'''Extracts the folder names for the stimuli used in a particular Layer 4 model.

	Parameters
	----------
	drive_path : string
		the filepath to access the Brain2017 data

	model : string
		the name of the Layer 4 model; either ll1, ll2, or ll3

	Returns
	-------
	stimuli : list
		a list of strings containing the stimuli folder names
	'''
	# create path to directory of stimuli
    stimulus_path = os.path.join(drive_path, 'layer4_spikes/simulations_' + model)
    stimuli_names = []
    # iterate over folders in the stimulus path and add their names to the list
    for files in os.listdir(stimulus_path):
        stimuli_names.append(files)
    return stimuli_names

def extract_outputs(drive_path, model, stimulus):
	'''Extracts the folder names for the simulated spike outputs obtained for a particular stimulus in a Layer 4 model.

	Parameters
	----------
	drive_path : string
		the filepath to access the Brain2017 data

	model : string
		the name of the Layer 4 model; either ll1, ll2, or ll3

	stimulus : string
		the name of the stimulus type to

	Returns
	-------
	output_list : list
		a list of strings containing the stimuli folder names
	'''
	# create path to directory of outputs, based on specific stimulus choice
    output_path = os.path.join(drive_path, 'layer4_spikes/simulations_' + model, stimulus)
    output_list = []
    # iterate over files in the output path and add the folders to the list
    for files in os.listdir(output_path):
    	# only files beginning with 'output' contain the spike information
        if files[:6] == 'output':
            output_list.append(files)
    return output_list

def get_spike_filepath(layer4_spikes, model, stimulus, stimulus_id, trial):
    '''Returns the filepath to the dat file containing the spike times from a given Layer 4 simulation.
    
    Parameters
    ----------
        layer4_spikes : string
            the filepath to the layer4_spikes folder
        
        model : string
            the simulation model, either 'll1', 'll2', or 'll3'
        
        stimulus : string
            the stimulus type, either 'bars', 'flashes', 'gratings', 'natural_images', 'natural_movies', or 'spont',
            depending on simulation model
        
        stimulus_id : int
            id tag identifying which particular image of the stimulus was used
        
        trial : int
            specifies the trial number, if necessary
    
    Returns
    -------
        filepath : string
            the filepath to the .dat file containing the spike times for the desired simulation
            IOError if file not found
    '''
    
    ### TODO : handle non-natural image case
    if stimulus == 'gratings':
        filepath = os.path.join(layer4_spikes, 'simulations_' + model, stimulus,
                                'output_%s_g%d_%d_sd278' % (model, stimulus_id, trial),
                                'spk.dat')
    elif stimulus == 'natural_images':
        filepath = os.path.join(layer4_spikes, 'simulations_' + model, stimulus, 
                                'output_%s_imseq_%d_%d_sd278' % (model, stimulus_id, trial),
                                'spk.dat')
    
    # check whether file exists
    if os.path.isfile(filepath):
        return filepath
    else:
        raise IOError('File not found.')

### Functions to interface with spiking data

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
