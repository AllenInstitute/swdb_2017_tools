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
        neuron_ids[idx] = neuron_id
        spikes = spike_times_per_neuron[neuron_id]
        num_spikes = len(spikes)
        # amount of padding needed
        padding = max_num_spikes - num_spikes 
        # add spikes along with padding to the spike times array
        spike_times[neuron_id, :] = np.lib.pad(spikes, (0, padding), 'constant', constant_values = (0, -1))
        
    return neuron_ids, spike_times

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
    num_bins = int((t_finish - t_start)/bin_size)
    bins = np.linspace(t_start, t_finish, num_bins)
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
    num_bins = int((t_finish - t_start)/bin_size)
    bins = np.linspace(t_start, t_finish, num_bins)
    spike_train = {neuron_id : np.histogram(spike_times_per_neuron[neuron_id], bins)[0] for neuron_id in spike_times_per_neuron}
    return spike_train

### functions below are specific to layer4 simulation data

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
    if stimulus == 'natural_images':
        filepath = os.path.join(layer4_spikes, 'simulations_' + model, stimulus, 
                                'output_%s_imseq_%d_%d_sd278' % (model, stimulus_id, trial),
                                'spk.dat')
    
    # check whether file exists
    if os.path.isfile(filepath):
        return filepath
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