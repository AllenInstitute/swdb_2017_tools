from scipy.stats import kendalltau as kt
import numpy as np
import scipy.spatial.distance
import matplotlib.pyplot as plt



def vectorize(mat):
    """Takes a square symmetric matrix mat and returns the vectorized form. Like matlab's squareform.

         Note: could probably also just use scipy's squareform function."""
    assert mat.shape[0] == mat.shape[1]

    vec = mat[:, 0]
    for row in range(1, mat.shape[1]):
        vec = np.concatenate((vec, mat[row:, row]))

    return vec


def get_kt(rsm1,rsm2):
    '''Gets Kendall tau-a measurements between two RDM matrices, first vectorizes matrices
    and then computes kt using scipy kendall-tau function'''
    #vecRDM1 = vectorize(RDM1)
    #vecRDM2 = vectorize(RDM2)
    vec_rsm1 = scipy.spatial.distance.squareform(rdm1)
    vec_rsm1 = scipy.spatial.distance.squareform(rdm2)
    k = kt(vec_rdm1, vec_rdm1).correlation
    return k