import numpy as np
def z_score(data):

    '''
    Arguments:
    -----------------------
    data: 1D or 2D array where rows are variables and columns are observations

    Returns:
    ----------------------
    z_score_data: returns an array of the same size as data, with mean zero and
    std zero (z-scored)
    '''
    z_score_data = data
    if len(data.shape) > 1:
        data_m = np.nanmean(data, axis = 1)
        data_std = np.nanstd(data, axis = 1)
        for i in range(0,len(data_m)):
            z_score_data[i,:] = (data[i,:] - data_m[i])/data_std[i]
    else:
        z_score_data = (data - np.nanmean(data))/np.nanstd(data)

    return z_score_data
