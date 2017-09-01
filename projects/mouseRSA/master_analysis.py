from scipy.stats import kendalltau as kt
import numpy as np
import scipy.spatial.distance

# We need to import these modules to get started
import numpy as np
import pandas as pd
import os
import sys
import h5py
import seaborn as sns 
from scipy.ndimage.measurements import center_of_mass

import matplotlib.pyplot as plt

drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def get_experiments_grouped(exps,targeted_structures,cre_lines,imaging_depths, stim_type='all',
                            rsm_name = 'rsa_mean_mean_mahala', cluster = None,
                            donor_name=None,rsm_selection='mean'):
    ''' Gets Kendall tau measurement for all matrices for a certain list of
    visual areas, cre lines, and imaging depths
    If multiple RSAs exist for one mouse, average them together
    If multiple RSAs for one targeted structure, cre line, or imaging depth 
    
    For now, takes a list of ei'''
    if cluster is not None:
        exps = exps.iloc[cluster,:]
        
        
    #  remove rows if the rsa == 0
    # note we have to do this stinky thing because just testing ==0 won't work for the RSA arrays
    exps = exps[ 
        ~exps[rsm_name].apply(
            lambda x: isinstance(x,int)).values]
    


    # first get 'maximal subset' of valid indexes: those strings present in all valid index lists

    maximal_subset = list(pd.Series(np.hstack(exps.valid_stim)).unique())
    for cre_line in cre_lines:
        imaging_depths=exps[exps.cre_line==cre_line].imaging_depth.unique()
        for imaging_depth in imaging_depths:
            for targeted_structure in targeted_structures:
                available_targeted_structures=exps[(exps.cre_line==cre_line) & 
                                                   (exps.imaging_depth==imaging_depth)].targeted_structure.unique()
                if targeted_structure in available_targeted_structures:
                    exp_group = exps.groupby(['cre_line','imaging_depth','targeted_structure']).get_group(
                            (cre_line,imaging_depth,targeted_structure))

                    
                    # loop to create subset list
                    for i in range(len(exp_group)):
                        for s in maximal_subset:
                            if s not in exp_group['valid_stim'].iloc[i]:
                                maximal_subset.remove(s)
                                continue
                    
    
    
    # now allow for just some stimuli types
        
    if stim_type=='static_gratings':
        for s in maximal_subset:
            if '(' not in i:
                maximal_subset.remove(s)
        #drop all else from 'valid_stim'
    elif stim_type=='natural_movie':
        for s in maximal_subset:
            if 'nm' not in i:
                maximal_subset.remove(s)
    elif stim_type=='natural_movie':
        for s in maximal_subset:
            if isinstance(s,str):
                maximal_subset.remove(s)            
        
    
    exps_grouped=pd.DataFrame(columns=['cre_line','imaging_depth','targeted_structure','rsm'])
    for cre_line in cre_lines:
        imaging_depths=exps[exps.cre_line==cre_line].imaging_depth.unique()
        for imaging_depth in imaging_depths:
            for targeted_structure in targeted_structures:
                available_targeted_structures=exps[(exps.cre_line==cre_line) & 
                                                   (exps.imaging_depth==imaging_depth)].targeted_structure.unique()
                if targeted_structure in available_targeted_structures:
                    
                    exp_group = exps.groupby(['cre_line','imaging_depth','targeted_structure']).get_group(
                            (cre_line,imaging_depth,targeted_structure))
                                        
                    rsms=exp_group[rsm_name]
                    
                    
                    #now subselect rows and columns
                        
                    
                    if rsm_selection=='mean':
                        rsms = rsms.apply(lambda x: x.loc[maximal_subset, maximal_subset].values)

                        rsm = rsms.mean()
                        rsm = pd.DataFrame(data = rsm, index = maximal_subset, columns = maximal_subset)
                        
                        to_add=pd.DataFrame(data=[cre_line,imaging_depth,targeted_structure,rsm],
                                            index=['cre_line','imaging_depth','targeted_structure','rsm'])
                    elif rsm_selection=='random':
                        rsms = rsms.apply(lambda x: x.loc[maximal_subset, maximal_subset].values)

                        rsm = rsms.iloc[np.random.randint(
                            0,high=exp_group.shape[0])]
                        rsm = pd.DataFrame(data = rsm, index = maximal_subset, columns = maximal_subset)


                        to_add=pd.DataFrame(data=[cre_line,imaging_depth,targeted_structure,rsm],
                                            index=['cre_line','imaging_depth','targeted_structure','rsm'])
                    elif rsm_selection=='all':
                        #first the first
                        rsms = rsms.apply(lambda x: x.loc[maximal_subset, maximal_subset])

                        to_add=pd.DataFrame(data=[cre_line,imaging_depth,targeted_structure, 
                                                  rsms.iloc[0]],
                                            index=['cre_line','imaging_depth','targeted_structure','rsm'])    
                        # now we add the others
                        for n in range(1,rsms.shape[0]):
                            to_add[str(n)]=pd.Series([cre_line,imaging_depth, targeted_structure,
                                                      rsms.iloc[n].loc[maximal_subset, maximal_subset]],
                                                     index=['cre_line','imaging_depth','targeted_structure','rsm'])
                            
                    exps_grouped=exps_grouped.append(to_add.T,ignore_index=True)
                    
                else:
                    continue
                    
        
    return remove_nan_rsms(exps_grouped)

def get_kt(rsm1,rsm2):
    '''Gets Kendall tau-a measurements between two RDM matrices, first vectorizes matrices
    and then computes kt using scipy kendall-tau function'''
    np.fill_diagonal(rsm1,0)
    np.fill_diagonal(rsm2,0)
    vec_rsm1 = vectorize(rsm1)
    #
    vec_rsm2 = vectorize(rsm2)

    #vec_rsm1 = scipy.spatial.distance.squareform(rsm1)
    #vec_rsm2 = scipy.spatial.distance.squareform(rsm2)
    k = kt(vec_rsm1, vec_rsm2).correlation
    return k

def get_kt_matrix(exps_grouped,compare):
    kt_dfs = []
    if compare=='targeted_structure':
        cre_lines=exps_grouped.cre_line.unique()
        for cre_line in cre_lines:
            imaging_depths=exps_grouped[exps_grouped.cre_line==cre_line].imaging_depth.unique()
            for imaging_depth in imaging_depths:
                exps_to_compare=exps_grouped.groupby(['cre_line','imaging_depth']).get_group((cre_line,imaging_depth))
                to_compare=exps_to_compare[compare]
                kt_matrix=np.zeros((len(to_compare),len(to_compare)))

                for i,rsm1 in enumerate(exps_to_compare.rsm):
                    for j,rsm2 in enumerate(exps_to_compare.rsm): 

                        kt_matrix[i,j]=get_kt(rsm1,rsm2)
                np.fill_diagonal(kt_matrix,0)
                kt_df=pd.DataFrame(data=kt_matrix,columns=to_compare,index=to_compare)
                fig=plt.figure()
                fig.add_subplot(111)
                kt_dfs.append(kt_df)
                sns.heatmap(kt_df,vmin=0)
                plt.title(cre_line+' '+str(imaging_depth))
                
    elif compare=='imaging_depth':
        cre_lines=exps_grouped.cre_line.unique()
        for cre_line in cre_lines:
            targeted_structures=exps_grouped[exps_grouped.cre_line==cre_line].targeted_structure.unique()
            for targeted_structure in targeted_structures:
                exps_to_compare=exps_grouped.groupby(['cre_line','targeted_structure']).get_group((cre_line,targeted_structure))
                to_compare=exps_to_compare[compare]
                kt_matrix=np.zeros((len(to_compare),len(to_compare)))

                for i,rsm1 in enumerate(exps_to_compare.rsm):
                    for j,rsm2 in enumerate(exps_to_compare.rsm): 
                        kt_matrix[i,j]=get_kt(rsm1,rsm2)
                np.fill_diagonal(kt_matrix,0)
                kt_df=pd.DataFrame(data=kt_matrix,columns=to_compare,index=to_compare)
                fig=plt.figure()
                fig.add_subplot(111)
                sns.heatmap(kt_df,vmin=0,vmax=0.5)
                plt.title(cre_line+' '+str(targeted_structure))
                
    return kt_dfs
        

def get_population_rf(boc, experiment_id):
    c_flag = 'C'
    lsn_name = 'locally_sparse_noise'
    rf_name = 'receptive_field_lsn'
    #
    for a in boc.get_ophys_experiments(experiment_container_ids=[experiment_id]):
        if a['session_type'].endswith('2'):
            c_flag = 'C2'
            if a['targeted_structure'] != 'VISp':
                lsn_name = 'locally_sparse_noise_8deg'
                rf_name = 'receptive_field_lsn8'
            else:
                lsn_name = 'locally_sparse_noise_4deg'
                rf_name = 'receptive_field_lsn4'

    drive_path = boc.manifest.get_path('BASEDIR')
    if c_flag=='C':
        session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C_analysis.h5')
    elif c_flag=='C2':    
        session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C2_analysis.h5')

    
    f = h5py.File(analysis_file, 'r')
    receptive_field = f['analysis'][rf_name].value
    f.close()
    pop_rf = np.nansum(receptive_field, axis=(2,3))
    return pop_rf


def get_rfs(exps):
    """Just loads the receptive field matrics and returns a list of them."""
    rfs = []
    for exp_id in exps.id.values:
    
        try:
            rf = get_population_rf(boc, exp_id)
        except: 
            rf.append(np.array([np.nan, np.nan]))
        rfs.append(rf)
    return rfs

def get_weighted_stdev(rf,com):
    factor = 9.3 if rf.shape== (8,14) else 4.6
    
    #first get distances
    ds =[]
    ws = []
    
    # normalize rf to 1
    rf = rf/rf.mean()
    
    for i in range(rf.shape[0]):
        for j in range(rf.shape[1]):
            ds.append(np.linalg.norm([i*factor-com[0],j*factor-com[1]]))
            ws.append(rf[i,j])
            
    std = np.std( np.array(ds) * np.array(ws) )
    
    return std


def get_valid_cluster(rfs, exps, center):
    """Returns a boolean array of length exps for this cluster. 
    exps = experiment dataframe
    rfs = list of rf arrays
    center = length 2. position."""
    # First get RF center of masses
    coms = []
    for rf in rfs:
        
        
        if rf.shape== (8,14):
                # 8 deg sparse noise
            com = np.array(center_of_mass(rf))*9.3 # to degrees
        elif rf.shape== (2,):
            com = np.array([np.nan, np.nan])
        else:
            com = np.array(center_of_mass(rf))*4.6 # to degrees


        coms.append(com)
        
    # now get distances to center
    center = np.array(center)
    dists = np.linalg.norm(np.array(coms)-center,axis=1)

    areas = exps.targeted_structure
    area_stds = {}
    # get average standard deviations: RMS distance from centroid
    for area in areas.unique():
        this_std = []
        # loop through all exps
        for i,rf in enumerate(rfs):
            if area != areas[i]: 
                continue
            if rf.shape== (2,): 
                this_std.append(np.nan)
            else:
                this_std.append(get_weighted_stdev(rf, coms[i]))
        area_stds[area] = np.nanmean(this_std)
        
    # create boolean array. True if < area's stdev
    valids = []
    for i in range(len(dists)):
        if dists[i] < area_stds[areas[i]]/2:
            valids.append(True)
        else: valids.append(False)
    
        
    return valids


def vectorize(mat):
    """Takes a square symmetric matrix mat and returns the vectorized form. Like matlab's squareform.

         Note: could probably also just use scipy's squareform function."""
    assert mat.shape[0] == mat.shape[1]

    vec = mat[:, 0]
    for row in range(1, mat.shape[1]):
        vec = np.concatenate((vec, mat[row:, row]))

    return vec

def remove_nan_rsms(exps_grouped):
    nan_indexes = []
    for i,r in exps_grouped.iterrows():
        rsm = r.rsm.values
        
        this_nan_indexes = np.argwhere(np.sum(np.isnan(rsm),axis=1)> rsm.shape[0]/2 ).flatten()
        nan_indexes += list(this_nan_indexes)
    
    nan_indexes = np.unique(nan_indexes)
    
    # now remove them
    for i,r in exps_grouped.iterrows():
        exps_grouped['rsm'][i] = r.rsm.drop(r.rsm.index[nan_indexes], axis = 0).drop(r.rsm.index[nan_indexes], axis = 1)
        
        
    return exps_grouped

def get_all_kt_matrix(exps_grouped, distance_metric = 'kt'):
    kt = np.zeros((len(exps_grouped),len(exps_grouped)))
    
    
    if distance_metric == 'kt':
        for i,rsm1 in enumerate(exps_grouped.rsm):
            for j,rsm2 in enumerate(exps_grouped.rsm): 
                if i<j:
                    kt[i,j]=get_kt(rsm1.values,rsm2.values)
            
        kt = np.triu(kt) + np.triu(kt, 1).T 
    else:
        resps = get_unfolded_rsm(exps_grouped)
        kt = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(resps, distance_metric))
        
    np.fill_diagonal(kt,np.nan)

    kt_df=pd.DataFrame(data=kt,columns=exps_grouped.targeted_structure,index=exps_grouped.targeted_structure)
    

                
    return kt_df

def get_all_kt_matrix_labeled(exps_grouped, distance_metric = 'kt'):
    kt = np.zeros((len(exps_grouped),len(exps_grouped)))
    
    
    if distance_metric == 'kt':
        for i,rsm1 in enumerate(exps_grouped.rsm):
            for j,rsm2 in enumerate(exps_grouped.rsm): 
                if i<j:
                    kt[i,j]=get_kt(rsm1.values,rsm2.values)
            
        kt = np.triu(kt) + np.triu(kt, 1).T 
    else:
        resps = get_unfolded_rsm(exps_grouped)
        kt = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(resps, distance_metric))
        
    columns=[]
    for i,r in exps_grouped.iterrows():
        string=(r.targeted_structure,r.cre_line, r.imaging_depth)
        columns.append(string)
    kt_df=pd.DataFrame(data=kt,columns=columns,index=columns)


                 
    return kt_df


def get_all_kt_multiframe_labeled(exps_grouped, distance_metric = 'kt'):
    kt = np.zeros((len(exps_grouped),len(exps_grouped)))
    
    
    if distance_metric == 'kt':
        for i,rsm1 in enumerate(exps_grouped.rsm):
            for j,rsm2 in enumerate(exps_grouped.rsm): 
                if i<j:
                    kt[i,j]=get_kt(rsm1.values,rsm2.values)
            
        kt = np.triu(kt) + np.triu(kt, 1).T 
    else:
        resps = get_unfolded_rsm(exps_grouped)
        kt = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(resps, distance_metric))
        
    columns=[]
    for i,r in exps_grouped.iterrows():
        string=(r.targeted_structure,r.cre_line, r.imaging_depth)
        columns.append(string)
        
        
        
    
    mi = pd.MultiIndex(levels=[targeted_structures,cre_lines,imaging_depths,],
           labels=[replace_str_w_index(zip(*columns)[0],targeted_structures),
                   replace_str_w_index(zip(*columns)[1],cre_lines),
                   replace_str_w_index(zip(*columns)[2],imaging_depths)],
           names=['Area', 'Cre', 'Depth'])

    kt_mi = pd.DataFrame(kt, index=mi, columns=mi)
    
                 
    return kt_mi

def replace_str_w_index(list_to_replace, reference_list):
    """Takes a list with values, and replaces those values with their position in reference list.
    For use in making MultiIndex"""
    
    final_list = np.zeros((len(list_to_replace)))
    for i, el in enumerate(reference_list):
        is_here = np.where(np.array(list_to_replace)==el, i,0)
        
        final_list += is_here
        
    return final_list
    
    


def get_unfolded_rsm(exps_grouped):
    """Takes the result of get_experiments_grouped and performs a 2D embedding on the rsms.
    Output: 2D numpy array of length number_experiments"""
    n_stim = exps_grouped.loc[0,'rsm'].shape[0]
    matrix_to_embed = np.vstack(
                        exps_grouped.rsm.apply( # apply this function to each rsm
                                scipy.spatial.distance.squareform))
        
    return matrix_to_embed


def scatter_viz(to_plot,exps_grouped):
    
    plt.subplots_adjust(bottom = 0.1)
    plt.subplots_adjust(bottom = 0.1)

    areas = exps_grouped.targeted_structure
    
    for area in areas.unique():

        plt.scatter(
            to_plot[areas.values==area, 0], to_plot[areas.values==area, 1], marker = 'o'
            )

    plt.legend(areas.unique())