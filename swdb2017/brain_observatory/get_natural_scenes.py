def get_natural_scenes(boc, exp_id):
    
    # Load the experimental data
    data_set = boc.get_ophys_experiment_data(exp_id)

    # Read in the array of images
    scenes = data_set.get_stimulus_template('natural_scenes')

    return scenes