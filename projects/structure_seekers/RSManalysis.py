## Representational similarity matrix analysis

#%% Set drive path
# OS X
drive_path = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'


#%% Imports!
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from swdb2017.brain_observatory import get_brain_observatory_expdata as gbed
from swdb2017.general_tools import create_dendrogram as dend
from swdb2017.general_tools import reoroder_representational_similarity as reorderrsm
import pylab
from scipy.cluster.hierarchy import dendrogram
import json
import scipy.stats as spstats

#%%
def exp_id_tool(number):
   if number > 199:
       return expIDs.index(number)
   else:
       return expIDs[number]

#%% Import dataset
# manifest_path is a path to the manifest file. We will use the manifest 
# file preloaded onto your Workshop hard drives. Make sure that drive_path 
# is set correctly for your platform. (See the first cell in this notebook.) 

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)

exps = pd.DataFrame(boc.get_experiment_containers())

# Representational similarity matrices with experiment and session ids

#rs = gbed.get_all_representational_similarity_matrices(boc,'three_session_B',
#                                                       'natural_scenes',drive_path)

rs_filename = '/Users/cschneider/Desktop/ColleenDesktop/Mahon Lab/Courses/SWDB/Project/repsimmat.json'
with open(rs_filename,'r') as data_file:
    rs = json.load(data_file)  #list of flattened arrays

#%% Z score each representational similarity matrix
rs_mat = np.array(rs['representational_similarity'])[:,1:119,1:119]
for i in range(199):
    np.fill_diagonal(rs_mat[i,:,:],np.nan)
rs_mat_z = np.zeros(rs_mat.shape)
for i in range(len(rs['representational_similarity'])):
    test_rs_mat = np.copy(rs_mat[i,:,:])
    test_rs_mat[~np.isnan(test_rs_mat)] = spstats.mstats.zscore(test_rs_mat[~np.isnan(test_rs_mat)])
    rs_mat_z[i,:,:] = np.copy(test_rs_mat)

#%% Category dendrogram

#load annotated image set. Image IDs are leaves on dendrogram
annotated_images = '/Users/cschneider/Desktop/ColleenDesktop/Mahon Lab/Courses/SWDB/Project/image_subleaf.csv'

anot_ims = pd.read_csv(annotated_images)


d = { 'Images': ['Living', 'Non-living'], 'Living': ['Animals', 'Plants'], 'Non-living':['Man-made','Scenes'],
                 'Animals':['bird','bug','mammal','reptile'], 'Plants':['flower','leaf','tree'],
                 'Man-made':['building','other'], 'Scenes':['ground','landscape'],
                 'landscape':['mountains','field','lake'],
                 'mountains':[],'field':[],'lake':[],
                 'mammal':['bear','cat','dog','monkey','antlers','elephant','other-mammal'],
                 'bear':[],'cat':['tiger','cheetah','leopard','bobcat','lion'],
                 'tiger':[],'cheetah':[],'leopard':[],'bobcat':[],'lion':[],
                 'dog':['fox','coyote','wolf','puppies'],
                 'fox':[],'coyote':[],'wolf':[],'puppies':[],
                 'monkey':[],'antlers':[],'elephant':[],'other-mammal':[],
                 'bird':['long-beak','flying','long-neck','nesting'],
                 'long-beak':[],'flying':[],'long-neck':[],'nesting':[],
                 'flower':['closeup','far'],
                 'closeup':[],'far':[],
                 'tree':['multiple','one','top','bottom'],
                 'multiple':[],'one':[],'top':[],'bottom':[],
                 'leaf':[],'building':[],'other':[],'ground':[],'reptile':[],'bug':[],
                 }

categories = pd.unique(anot_ims['category'])
for cat in categories: #make categories inner nodes instead of leaves
    d[cat].extend(pd.Series.tolist(anot_ims['Image'][anot_ims['category']==cat]))
    
for i in range(len(anot_ims)): #add leaves
    d[anot_ims['Image'][i]] = []

Z = dend.create_dendrogram(d) #creates dendrogram from specified nodes and trees in d
#%% plot matrices ordered by dendrogram

targeted_structure = pd.DataFrame([exps['id'].values, exps['targeted_structure'].values]).T
targeted_structure.columns = ['experiment_ids','struct']
structures = pd.Series.tolist(targeted_structure.struct)
expIDs = pd.Series.tolist(targeted_structure.experiment_ids)


rs_reorder = reorderrsm.reorder_representational_similarity_plot(Z,199,rs_mat_z,expIDs,structures,'_RawTrue_exp')


