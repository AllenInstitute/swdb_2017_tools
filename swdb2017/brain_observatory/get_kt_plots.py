
# We need to import these modules to get started
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%matplotlib inline

from scipy.stats import kendalltau as kt

import scipy.spatial.distance

'''All of these functions take DataFrame exps, which is just like a what you get from a get_ophys_experiments object
out of the brain observatory cache, except populated with RSMs. Only designed to work with one type of stimulus (natural scenes) 
at the moment'''
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
    np.fill_diagonal(rsm1,0)
    np.fill_diagonal(rsm2,0)
    vec_rsm1 = vectorize(rsm1)
    #
    vec_rsm2 = vectorize(rsm2)

    #vec_rsm1 = scipy.spatial.distance.squareform(rsm1)
    #vec_rsm2 = scipy.spatial.distance.squareform(rsm2)
    k = kt(vec_rsm1, vec_rsm2).correlation
    return k

def group_imaging_depths(exps):
    '''Groups imaging depth by layer, approximately'''
    exps.loc[(exps.imaging_depth > 100) & (exps.imaging_depth < 200),'imaging_depth'] = 175
    exps.loc[(exps.imaging_depth > 200) & (exps.imaging_depth <= 300), 'imaging_depth'] = 275
    exps.loc[(exps.imaging_depth > 300) & (exps.imaging_depth < 400), 'imaging_depth'] = 350
    exps.loc[(exps.imaging_depth > 400) & (exps.imaging_depth < 500), 'imaging_depth'] = 435


def get_experiments_grouped(exps, targeted_structures, cre_lines, imaging_depths,
                            rsm_selection='random'):
    ''' Returns pandas dataframe of list of experiments from the given targeted structures, cre-lines, and imaging depths
    with RSM constructed by averaging all RSMs from given parameters, or a random RSM. '''
    if type(imaging_depths) != 'list':
        imaging_depths = [imaging_depths]
    if type(cre_lines) != 'list':
        cre_lines = [cre_lines]

    exps_grouped = pd.DataFrame(columns=['cre_line', 'imaging_depth', 'targeted_structure', 'rsm'])
    for cre_line in cre_lines:
        for imaging_depth in imaging_depths:
            for targeted_structure in targeted_structures:
                available_targeted_structures = exps[
                    (exps.cre_line == cre_line) & (exps.imaging_depth == imaging_depth)].targeted_structure.unique()
                if targeted_structure in available_targeted_structures:
                    if rsm_selection == 'mean':
                        rsm = exps.groupby(['cre_line', 'imaging_depth', 'targeted_structure']).get_group(
                            (cre_line, imaging_depth, targeted_structure)).rsm.mean()
                    elif rsm_selection == 'random':
                        rsm = exps.groupby(['cre_line', 'imaging_depth', 'targeted_structure']).get_group(
                            (cre_line, imaging_depth, targeted_structure)).rsm.iloc[np.random.randint(0, high=
                        exps.groupby(['cre_line', 'imaging_depth', 'targeted_structure']).get_group(
                            (cre_line, imaging_depth, targeted_structure)).shape[0])]
                    to_add = pd.DataFrame(data=[cre_line, imaging_depth, targeted_structure, rsm],
                                          index=['cre_line', 'imaging_depth', 'targeted_structure', 'rsm'])
                    exps_grouped = exps_grouped.append(to_add.T, ignore_index=True)
                else:
                    continue
    return exps_grouped


def get_kt_matrix(exps_grouped, compare):
    '''Gets Kendall tau matrices for the experiments specified in exps_grouped'''
    to_compare = exps_grouped[compare]
    kt_matrix = np.zeros((len(to_compare), len(to_compare)))
    for i, rsm1 in enumerate(exps_grouped.rsm):
        for j, rsm2 in enumerate(exps_grouped.rsm):
            kt_matrix[i, j] = get_kt(rsm1, rsm2)
    np.fill_diagonal(kt_matrix, 0)
    kt_df = pd.DataFrame(data=kt_matrix, columns=to_compare, index=to_compare)
    return kt_df



###############################

exps=group_imaging_depths(exps)

cre_lines=boc.get_all_cre_lines()
imaging_depths=boc.get_all_cre_lines()
targeted_structures=boc.get_all_targeted_structures()

for cre_line in cre_lines:
    imaging_depths = exps[exps.cre_line == cre_line].imaging_depth.unique()
    for imaging_depth in imaging_depths:
        exps_grouped = get_experiments_grouped(exps, targeted_structures, cre_line, imaging_depth)
        kt_df = get_kt_matrix(exps_grouped, 'targeted_structure')
        fig = plt.figure()
        fig.add_subplot(111)
        sns.heatmap(kt_df, vmin=0, vmax=0.125)
        plt.title(cre_line + ' ' + str(imaging_depth))

