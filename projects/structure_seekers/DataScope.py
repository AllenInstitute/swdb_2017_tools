## Organize Scope of the Data and plot a dendrogram

#%% Set drive path

# OS X
drive_path = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache'


#%% Imports!
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from swdb2017.general_tools import create_dendrogram as dend
from scipy.cluster.hierarchy import dendrogram
from collections import Counter

#%% Import dataset
# manifest_path is a path to the manifest file. We will use the manifest 
# file preloaded onto your Workshop hard drives. Make sure that drive_path 
# is set correctly for your platform. (See the first cell in this notebook.) 

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
boc = BrainObservatoryCache(manifest_file=manifest_file)


#%% Load Specific Data
exps = pd.DataFrame(boc.get_experiment_containers())


#%% Create dendrogram to determine the number of experiments in each leaf

struct_branches = exps.targeted_structure.unique()
struct_line_branches = []
full_branches = []
for exp in range(exps.shape[0]):
    full_branches.append(exps.targeted_structure[exp] +'__' + exps.cre_line[exp] + '__' + str(exps.imaging_depth[exp]))
    struct_line_branches.append(exps.targeted_structure[exp] +'__' + exps.cre_line[exp])
struct_line_branches = np.array(struct_line_branches)
full_branches = np.array(full_branches)

d = {'Visual Areas':np.ndarray.tolist(pd.unique(exps.targeted_structure))}
for i in range(len(exps)):
    if d.has_key(exps.targeted_structure[i])==False:
        d[exps.targeted_structure[i]] = [struct_line_branches[i]]
    elif not struct_line_branches[i] in d[exps.targeted_structure[i]]:
        d[exps.targeted_structure[i]].append(struct_line_branches[i])
for i in range(len(exps)):
    if d.has_key(struct_line_branches[i])==False:
        d[struct_line_branches[i]] = [full_branches[i]]
    elif not full_branches[i] in d[struct_line_branches[i]]:
        d[struct_line_branches[i]].append(full_branches[i])
for i in range(len(pd.unique(full_branches))):
    d[pd.unique(full_branches)[i]] = []
    
nodes = ['Visual Areas'] + np.ndarray.tolist(pd.unique(full_branches)) + np.ndarray.tolist(pd.unique(struct_branches)) + np.ndarray.tolist(pd.unique(struct_line_branches))
leaves = set(np.sort(np.ndarray.tolist(pd.unique(full_branches))))
inner_nodes = ['Visual Areas'] + np.ndarray.tolist(pd.unique(struct_line_branches)) + np.ndarray.tolist(pd.unique(struct_branches))

#%% Plot dendrograom
plt.figure()
Z = dend.create_dendrogram(d,nodes = nodes, leaves = leaves, inner_nodes = inner_nodes)

#%%
R = dendrogram(Z)
plt.figure(figsize = (15,5))
labels = np.sort(pd.unique(full_branches))
labelslist = [labels[i] for i in [R['leaves']]]
c1 = ['#990000','#FF0000','#FF66CC','#990099','#9933FF','#660099']
c2 = []
colors = ['k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k',
          'k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k',
          'k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k',
          'k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k','k',
          'k','k','k','k','k','k','k','k','k','k','k','k','k',
          c1[5],c1[5],c1[5],c1[5],
          c1[4],c1[4],c1[4],c1[4],
          c1[0],c1[0],c1[0],c1[0],
          c1[1],c1[1],c1[1],c1[1],
          c1[3],c1[3],c1[3],c1[3],
          c1[2],c1[2],c1[2],c1[2],c1[2],
          'w','w','w','w','w']
R = dendrogram(
    Z,
    labels = labels,
    link_color_func=lambda k: colors[k],
    leaf_rotation=90.,
    leaf_font_size=12.,
    #show_contracted=True,  # to get a distribution impression in truncated branches
    show_leaf_counts = True
)

plt.ylim((0,15))

