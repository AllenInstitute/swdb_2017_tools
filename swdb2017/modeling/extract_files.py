import os 

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
