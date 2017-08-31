import numpy as np
import matplotlib.pyplot as plt

def pearson_corr_coeff(x, y):
    '''
    Returns pearson correlation coefficient using numpy.corrcoef
    Note, if there are Nan in either vector, the corresponding indices will be
    removed in each array

    Parameters:
    ----------
    x: array
        1D numpy array
    y: array
        1D numpy array (dims must agree with x)

    Returns:
    --------
    corr_coef: int
        The pearson correlation coefficient between x and y
    '''

    x_inds = np.where(np.isnan(x))[0]
    y_inds = np.where(np.isnan(y))[0]
    nan_inds = np.sort(np.unique(np.concatenate((x_inds, y_inds))))
    x = np.delete(x, nan_inds)
    y = np.delete(y, nan_inds)

    corr_mat = np.corrcoef(x, y)
    corr_coef = corr_mat[1, 0]

    return corr_coef

def corr_matrix(arr_list):
    """
    Returns a matrix of pearson correlation coefficients between arrays

    Parameters:
    ----------
    arr_list : list
        list containing numpy arrays to be correlated

    Returns:
    -------
    coef_mat : matrix
        matrix of pearson correlation coefficients
    """
    # Create matrix
    nVar = len(arr_list)
    coef_mat = np.zeros((nVar, nVar))

    # Put correlation values into matrix
    for i in range(nVar):
        for j in range(i, nVar):
            coef_mat[i, j] = pearson_corr_coeff(arr_list[i], arr_list[j])

    # Create square matrix
    coef_mat = np.triu(coef_mat, 1).T + coef_mat

    return coef_mat


def get_corr_matrix(dictionary, figure=True):
    """
    Retrieves correlation matrix from a dictionary of variables and values to be correlated
    Optional to plot graphical representation of correlation matrix

    Parameters:
    ----------
    dictionary : key, value pairs
        keys : list of strings with names of variables
        values : list of arrays to be correlated

    Returns:
    -------
    
    """
    vals = dictionary.values()
    keys = dictionary.keys()

    coef_mat = corr_matrix(vals)

    if figure is True:
        plt.imshow(coef_mat, cmap='viridis', clim=(-1., 1.))
        plt.grid(False)
        plt.colorbar()
        plt.xticks(range(len(keys)), keys, rotation=45)
        plt.yticks(range(len(keys)), keys, rotation='horizontal')

    return coef_mat


def get_correlations_from_features(features, exp_id, figure=True):
    """
    Parameters:
    ----------
    features: list
        Available inputs(str): 'pupil_size', 'pupil_rate', 'saccade_rate', 'running_rate', 'running_speed'
        """
    dataset = boc.get_ophys_experiment_data(exp_id)
    dict_values = []
    dict_keys = []

    for i in features:

        if 'pupil_area' == i:
            pupil_area, _ = extract_smooth_pupil(dataset, sigma=4)
            dict_values.append(pupil_area)
            dict_keys.append('pupil_area')


        elif 'pupil_diameter' == i:
            _, pupil_diameter = extract_smooth_pupil(dataset, sigma=4)
            dict_values.append(pupil_diameter)
            dict_keys.append('pupil_diameter')

        elif 'pupil_rate' == i:
            pupil_rate = extract_smooth_pupil_rate(dataset, sigma=4)
            dict_values.append(pupil_rate)
            dict_keys.append('pupil_rate')

        elif 'saccade_rate' == i:
            saccade_rate = extract_smooth_saccade_rate(dataset, sigma=4)
            dict_values.append(saccade_rate)
            dict_keys.append('saccade_rate')

        elif 'running_rate' == i:
            running_rate = extract_smooth_running_rate(dataset, sigma=2)
            dict_values.append(running_rate)
            dict_keys.append('running_rate')

        elif 'running_speed' == i:
            running_speed = extract_smooth_running_speed(dataset, sigma=2)
            dict_values.append(running_speed)
            dict_keys.append('running_speed')

        else:
            print 'Not a valid feature'

    feature_dict = dict(zip(dict_keys, dict_values))

    corr_matrix = get_corr_matrix(feature_dict, figure = figure)

    return corr_matrix