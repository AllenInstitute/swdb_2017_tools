import numpy as np

def pearson_corr_coeff(x, y):

    '''
    Arguments:
    ---------------------------
    x: 1D or 2D numpy array
    y: 1D or 2D numpy array (dims must agree with x)

    Note, if there are Nan in either vector, the corresponding indices will be
    removed in each array

    Returns:
    ---------------------------
    c: The pearson correlation coefficient between x and y

    '''
    x = np.squeeze(x.reshape(-1,1))
    y = np.squeeze(y.reshape(-1,1))
    x_inds = np.where(np.isnan(x))[0]
    y_inds = np.where(np.isnan(y))[0]
    nan_inds = np.sort(np.unique(np.concatenate((x_inds, y_inds))))
    x = np.delete(x, nan_inds)
    y = np.delete(y, nan_inds)
    x_diffs = np.ediff1d(x)
    y_diffs = np.ediff1d(y)
    num = np.dot(x_diffs,y_diffs)
    den = (np.sqrt(sum(x_diffs**2)*sum(y_diffs**2)))
    c = num/den

    return c
