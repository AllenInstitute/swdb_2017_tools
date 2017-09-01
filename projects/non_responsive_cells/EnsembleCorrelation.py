"""
EnsembleCorrelation.py

A small part of the Non Visually Responsive Cell Project
Dynamic Brain Workshop, Friday Harbor WA
August 29, 2017
Jacob Portes, j.portes@columbia.edu

The following code implements the ensemble correlation algorithm outlined in 
the paper Visual stimuli recruit intrinsically generated cortical ensembles 
Jae-eun Kang Miller, Inbal Ayzenshtat, Luis Carrillo-Reid, and Rafael Juste 
(PNAS 2014). The following language is taken from this paper.

The similarity between ensembles was evaluated using: (1) Pearson correlation r
(2) Fisher transform of r

A threshold for significant correlation was established for each pairwise 
comparison. Establishing a threshold for each comparison is important because 
in binary data the number of active neurons in a frame influences a correlation
 coefficient between a pair of frames.
 
1. We generate 1,000 independent surrogate ensembles by randomizing active 
cells while preserving the number of active cells per frame in ONE of the 
frames in each comparison (shuffling across cells).

2. The threshold corresponding to a significance level of P less than 0.05 was 
estimated as the correlation coefficient that exceeded only 5% of 
correlation coefficients between these surrogate ensembles 
(using np.percentile)
"""

# AWS
# drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'

# We need to import these modules to get started
import numpy as np
import pandas as pd
import timeit
#import os
#import sys
#import h5py


import matplotlib.pyplot as plt

## Generate Threshold for Each Ensemble Comparison
# for a single comparison, generate surrogate by shuffling
def generate_threshold_surrogate(ens_a,ens_b,permute_num):

    surr_corr = np.zeros(permute_num)

    for k in range(permute_num):
        tmp_perm = np.random.permutation(ens_b) # permute one ensemble while keeping other stable
        r = np.corrcoef(ens_a,tmp_perm)[0,1] # Pearson Correlation
        surr_corr[k] = 0.5*np.log((1+r)/(1-r)) # Fisher z transform
    
    thresh_final = np.percentile(surr_corr,95) # percentile function
            
    return thresh_final # return bound for Fisher z transform


# So I think I have a good way to pull out all the combinations 
# by using meshgrid and taking the upper triangle

def EnsembleCorrelations(ensemble_array,surr_num):
    
    '''
    Parameters
    -----------
    ensemble_array: numpy array
        [cells, ensembles (binarized, ordered)]
    
    surr_num: int
        Number of desired surrogate ensembles
        
    Returns
    -----------
    ensemble_matrix:
        Return matrix of ensemble pairs above threshold
        size rows-->cells, cols-->ensembles, but in pairs!
    above_thresh_ensembles_boot: 
        Indices of coupled pairs, numpy arrayof pair indices (2 columns)
    z_values_boot:
        Fisher-z values of selected ensembles
    '''

    cell_len = ensemble_array.shape[0]
    num_ensembles = ensemble_array.shape[1]
    
    # Create meshgrid
    tmp_lin = np.linspace(0,num_ensembles-1,num_ensembles) # 0 1 2 3...num_ensembles-1
    x_ind, y_ind = np.meshgrid(tmp_lin,tmp_lin)
    # Get indices of upper triangle, ignore central diagonal. This gives (1,0) (1,1) (1,2) etc.
    ind_triu1 = np.triu_indices(num_ensembles,1) 
    
    # Choose value pairs in upper triangle
    xv = x_ind[ind_triu1].astype(int) # make sure these are ints and not floats
    yv = y_ind[ind_triu1].astype(int) 
    
    # Combine them into tuples
    xvyv_zip = zip(xv,yv) 
    
    # Calculate Pearson Correlation Coefficient
    R = [np.corrcoef(ensemble_array[:,i],ensemble_array[:,j])[0,1] for i,j in zip(xv,yv)]
    R = np.array(R)
    
    # Fisher transform all the correlation coefficients
    F = 0.5*np.log((1+R)/(1-R)) # what if R == 1?

    
    start_time = timeit.default_timer()
    
    ## Now Bootstrap with surrogate datasets!
    # Generate threshold for each comparison and store
    F_lower_thresh = [generate_threshold_surrogate(ensemble_array[:,i],ensemble_array[:,j],surr_num) for i,j in zip(xv,yv)]
    
    elapsed = timeit.default_timer() - start_time
    print('Elapsed surrogate time (s): ', elapsed)
    
    z_values_boot = F[F > F_lower_thresh]
    thresh_ind_boot = np.where(F > F_lower_thresh)[0] # note that this is 1D
    above_thresh_ensembles_boot = np.array(xvyv_zip)[[thresh_ind_boot]]
    
    # Flatten pairs that are above threshold
    tmp_flatten = above_thresh_ensembles_boot.flatten()
    
    # Pull out ensemble pairs, save in matrix
    ensemble_matrix = np.zeros((cell_len,len(tmp_flatten)))
    for i in range(len(tmp_flatten)):
        ensemble_matrix[:,i] = ensemble_array[:,tmp_flatten[i]]
        
    # Return matrix of ensembles above threshold
    # size rows-->cells, cols-->ensembles
    # Return indices of coupled pairs, tuple of lists
    # Return Fisher-z values of selected ensembles
    return ensemble_matrix, above_thresh_ensembles_boot, z_values_boot


    # Return all Fisher z values
    # return F
    
    # Return bootstrapped thresholds for all pair comparisons
    # return F_lower_thresh




