import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import swdb2017.brain_observatory.behavior.extract_pupil_features as epf
import swdb2017.brain_observatory.behavior.extract_running_features as erf
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

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

    x_no_nan = np.delete(x, nan_inds)
    y_no_nan = np.delete(y, nan_inds)

    corr_mat = np.corrcoef(x_no_nan, y_no_nan)
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
    corr_mat = np.zeros((nVar, nVar))

    # Put correlation values into matrix
    for i in range(nVar):
        for j in range(i, nVar):
            corr_mat[i, j] = pearson_corr_coeff(arr_list[i], arr_list[j])

    # Create square matrix
    corr_mat = np.triu(corr_mat, 1).T + corr_mat

    return corr_mat


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
        plt.imshow(coef_mat, cmap = 'plasma', clim = (-1., 1.))
        plt.grid(False)
        plt.colorbar()
        plt.xticks(range(len(keys)), keys, rotation = 'vertical')
        plt.yticks(range(len(keys)), keys, rotation = 'horizontal')

    return coef_mat


def get_correlations_from_features(boc, features, exp_id, figure=True):
    """
    Parameters:
    ----------
    features: list
        Available inputs(str): 'pupil_area_smooth', 'pupil_area_rate',
        'saccade_rate', 'running_rate_smooth', 'running_speed_smooth'
    exp_id : int
        Experiment session id
    figure : boolean
        True to return correlation matrix and plot, False to only return correlation matrix

    Returns:
    -------
    feature_corr_df : pandas dataframe
        contains pearson correlation coefficients split by behavior features being correlated (column and row)
    figure : plt
        Graphical representation of correlation matrix
        """
    # Load experiment data
    dataset = boc.get_ophys_experiment_data(exp_id)
    dict_values = []
    dict_keys = []

    # Pull out desired behavior features
    for i in features:
        if 'pupil_area_smooth' == i:
            pupil_area, _ = epf.extract_smooth_pupil(dataset, sigma=4)
            dict_values.append(pupil_area)
            dict_keys.append('pupil_area')

        elif 'pupil_area_rate' == i:
            pupil_area_rate, _ = epf.extract_smooth_pupil_rate(dataset, sigma=4)
            dict_values.append(pupil_area_rate)
            dict_keys.append('pupil_area_rate')

        elif 'saccade_rate' == i:
            saccade_rate = epf.extract_smooth_saccade_rate(dataset, sigma=4)
            dict_values.append(saccade_rate)
            dict_keys.append('saccade_rate')

        elif 'running_rate_smooth' == i:
            running_rate = erf.extract_smooth_running_rate(dataset, sigma=2)
            dict_values.append(running_rate)
            dict_keys.append('running_rate')

        elif 'running_speed_smooth' == i:
            running_speed = erf.extract_smooth_running_speed(dataset, sigma=2)
            dict_values.append(running_speed)
            dict_keys.append('running_speed')

        else:
            print('Invalid feature.')

    # Create dictionary of feature keys and values (np.arr)
    feature_dict = dict(zip(dict_keys, dict_values))

    # Create matrix of pearson correlation coefficients
    corr_matrix = get_corr_matrix(feature_dict, figure=figure)

    feature_corr_df = pd.DataFrame(corr_matrix, columns=[feature_dict.keys()], index=[feature_dict.keys()])

    return feature_corr_df
