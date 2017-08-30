import numpy as np

def get_roi_centroids(dataset):
    """Extract Cell coords from roi masks

    Parameters
    ----------
    dataset : BrainObservatoryNwbDataSet
        Allen Institute experiment file

    Returns
    -------
    out : numpy.ndarray
      An Nx2 array of tuples denoting x,y coordinates of the cells.
    """
    out = []
    for mask in dataset.get_roi_mask_array():
        y,x = mask.nonzero()
        out.append([y.mean(),x.mean()])

    return np.array(out)
