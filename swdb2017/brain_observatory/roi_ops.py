def get_roi_centroids(dataset):
    """Extract Cell coords from roi masks

    Parameters
    ----------
    dataset : BrainObservatoryNwbDataSet
        Allen Institute experiment file

    Returns
    -------
    array
      An array of tuples denoting x,y coordinates of the cell.
    """
    out = []
    for mask in dataset.get_roi_mask_array():
        y,x = mask.nonzero()
        out.append((y.mean(),x.mean()))

    return out
