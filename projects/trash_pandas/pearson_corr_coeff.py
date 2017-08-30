import numpy as np

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
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_inds = np.where(np.isnan(x))[0]
    y_inds = np.where(np.isnan(y))[0]
    nan_inds = np.sort(np.unique(np.concatenate((x_inds, y_inds))))

    x_no_nan = np.delete(x, nan_inds)
    y_no_nan = np.delete(y, nan_inds)

    corr_mat = np.corrcoef(x_no_nan, y_no_nan)
    corr_coef = corr_mat[1, 0]

    return corr_coef
