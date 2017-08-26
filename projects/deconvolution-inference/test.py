# Test functions
# Import some sample data

import numpy as np
import pandas as pd
import os
import sys
import h5py
import matplotlib.pyplot as plt
import ca_tools as tools

drive_path = '/Volumes/Brain2017/data/dynamic-brain-workshop/brain_observatory_cache/'
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
print manifest_file

boc = BrainObservatoryCache(manifest_file=manifest_file)

visual_area = 'VISp'
cre_line ='Cux2-CreERT2'

exps = boc.get_experiment_containers(targeted_structures=[visual_area], cre_lines=[cre_line])
pd.DataFrame(exps)
expt_container_id = np.random.choice(exps)['id']
print expt_container_id

exps = boc.get_experiment_containers(targeted_structures=[visual_area], cre_lines=[cre_line])
pd.DataFrame(exps)
expt_container_id = np.random.choice(exps)['id']
print expt_container_id
session_id = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id], stimuli=['natural_scenes'])[0]['id']
data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)

fl_trace = data_set.get_dff_traces()

ind = 1
x = fl_trace[0] # dereference the time
y = fl_trace[1][ind] # dereference the fl trace 

out = tools.ca_deconvolution(y)

print(out)