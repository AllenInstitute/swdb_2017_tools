def pearson_corr_coeff(x, y= None):

    if y != None:
        x_diffs = np.ediff1d(x)
        y_diffs = np.ediff1d(y)
        num = np.dot(x_diffs,y_diffs)
        den = (np.sqrt(sum(x_diffs**2)*sum(y_diffs**2)))
        c = num/den

    else:
        x_diffs = np.ediff1d(x)
        y_diffs = np.ediff1d(x)
        num = np.dot(x_diffs,y_diffs)
        den = (np.sqrt(sum(x_diffs**2)*sum(y_diffs**2)))
        c = num/den
    return c
