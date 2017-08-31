
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from swdb2017.general_tools import create_dendrogram as dend
import pylab
from scipy.cluster.hierarchy import dendrogram

def reorder_representational_similarity_plot(Z,n_figs,rs,savename):
    '''
    Parameters
    ----------
    Z : list
        linkage from scipy.cluster.hierarchy.linkage
    n_figs : int
        number of unique figures to create, usually corresponds to the number 
        of representational similarity matrices
    rs : numpy array
        representational similarity matrices. If more then one they should be 
        stacked in the 0th dimension
    savename : string
        savename will have an integer appended to the end corresponding to the
        index of the corresponding representational similarity matrix
        
    Returns
    -------
    plot(s) of similarity matrices ordered by the dendrogram given by linkage Z
    
    rs_reorder : numpy array
        the reordered similarity matrices based on the dendrogram given by
        linkage z
        
    Written by Colleen Schneider 8/30/17
    
    '''
    
    rs_reorder = np.zeros([n_figs,118,118])
    for i in range(n_figs):
        test_rs = rs[i,:,:]
        fig = pylab.figure(figsize=(8,8))
        ax1 = fig.add_axes([0.09,0.1,0.2,0.6])
        R = dendrogram(Z,orientation='left')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Compute and plot second dendrogram.
        ax2 = fig.add_axes([0.3,0.71,0.6,0.2])
        R = dendrogram(Z)
        ax2.set_xticks([])
        ax2.set_yticks([])
                
        idx1 = R['leaves']
        test_rs = test_rs[idx1,:]
        test_rs = test_rs[:,idx1]
        
        rs_reorder[i,:,:] = test_rs
        
        axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
        im = axmatrix.matshow(test_rs, aspect='auto', origin='lower', cmap=pylab.cm.viridis,vmin = -0.1, vmax = 0.3)
        axcolor = fig.add_axes([0.94,0.1,0.02,0.6])
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
        pylab.colorbar(im,cax = axcolor)
        
        figname = savename + str(i)
        figpath = '/Users/cschneider/Desktop/ColleenDesktop/Mahon Lab/Courses/SWDB/Project/RepSimMats'
        figfullfile = os.path.join(figpath,figname)
        pylab.savefig(figfullfile, bbox_inches='tight')
        
        pylab.close
        
    return rs_reorder