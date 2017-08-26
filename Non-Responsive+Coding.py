
# coding: utf-8

# ## Non-Responsive Coding

# Evironment Set-up

# In[1]:


# AWS
drive_path = '/data/dynamic-brain-workshop/brain_observatory_cache/'


# In[2]:


# We need to import these modules to get started
import numpy as np
import pandas as pd
import os
import sys
import h5py

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[3]:


from allensdk.core.brain_observatory_cache import BrainObservatoryCache

manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
print manifest_file

boc = BrainObservatoryCache(manifest_file=manifest_file)


# In[4]:


# Download a list of all targeted areas
targeted_structures = boc.get_all_targeted_structures()
print 'all targeted structures: ' + str(targeted_structures)
# Download a list of all stimuli
stims = boc.get_all_stimuli()
print 'all stimuli: ' + str(stims)


# In[5]:


# Create a dictionary depicting Cre lines
cre_to_layer={} #dic of cre line paired with layer of expression
cre_to_layer['Cux2-CreERT2']='Layer 2,3 & 4'
cre_to_layer['Emx1-IRES-Cre']='Pan excitatory expression'
cre_to_layer['Nr5a1-Cre']='Layer 4'
cre_to_layer['Rbp4-Cre_KL100']='Layer 5'
cre_to_layer['Rorb-IRES2-Cre']='Layer 4'
cre_to_layer['Scnn1a-Tg3-Cre']='Layer 4'
print(cre_to_layer)
cre_lines=cre_to_layer.keys()
print(cre_lines)


# In[6]:


# layer at different imaging depths varies per animal
# Dictionary below creates an approximate imaging depth to cortical layer # conversion

depth_to_layer={} #dic translating imaging depth to lahyer
depth_to_layer['175']='Layer 2/3'
depth_to_layer['265']='Layer 2/3'
depth_to_layer['275']='Layer 2/3'
depth_to_layer['300']='Layer 3'
depth_to_layer['320']='Layer 4'
depth_to_layer['325']='Layer 4'
depth_to_layer['335']='Layer 4'
depth_to_layer['350']='Layer 4'
depth_to_layer['365']='Layer 4'
depth_to_layer['375']='Layer 5'
depth_to_layer['435']='Layer 5'
print(depth_to_layer)


# ## Import Data from Visual Area and Cre-Line

# In[7]:


# Choose a visual area and Cre-line to being
visual_area = 'VISp'
cre_line ='Rbp4-Cre_KL100'

# Import data from targeted area and Cre-line
exps = boc.get_experiment_containers(targeted_structures=[visual_area], cre_lines=[cre_line])

#Create Data Frame
pd.DataFrame(exps)


# In[8]:


# Donor name is id linked to mouse
# Choose an experiment from 'id' column to begin
expt_container_id=555040113


# In[9]:


# Get information from experimental session for id from id container
expt_session_info = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id])

# This returns a list of dictionaries with information regarding experimental sessions of our specified exp container
print(expt_session_info)

# There are three sessions in every container that correspond to the 3 1 hr imaging session types


# In[10]:


# Create Data Frame of experimental sessions in your container
expt_session_info_df = pd.DataFrame(expt_session_info) #create pd dataframe so can look at each diff session from each expt
expt_session_info_df


# In[11]:


# Create a dictionary with session id's for all conditions
#First find id's for sessions A-C
session_id_A=expt_session_info_df[expt_session_info_df.session_type=='three_session_A'].id.values[0]
session_id_B=expt_session_info_df[expt_session_info_df.session_type=='three_session_B'].id.values[0]

# Note session_id_C needs to be adjusted based on whether C2 or C session type
session_id_C=expt_session_info_df[expt_session_info_df.session_type=='three_session_C2'].id.values[0]

# Create Session Dictionary
session_ids={'session_id_A':session_id_A, 'session_id_B':session_id_B, 'session_id_C':session_id_C}
print(session_ids)


# In[12]:


# Make a data_set object for each Session A-C
data_set_A = boc.get_ophys_experiment_data(ophys_experiment_id=session_id_A)
data_set_B = boc.get_ophys_experiment_data(ophys_experiment_id=session_id_B)
data_set_C = boc.get_ophys_experiment_data(ophys_experiment_id=session_id_C)
data_sets=[data_set_A, data_set_B, data_set_C]


# In[13]:


# Get the metadata for Sessions A-C
mdA = data_set_A.get_metadata()
mdB = data_set_B.get_metadata()
mdC = data_set_C.get_metadata()
print (mdA)
print (mdB)
print (mdC)


# In[14]:


#Get flourescene traces with neuropil correction
A=data_set_A.get_dff_traces()
B=data_set_B.get_dff_traces()
C=data_set_C.get_dff_traces()


# In[15]:


# Get timestamps by returning first part of tuple
timestamps_A=A[0]
timestamps_B=B[0]
timestamps_C=C[0]


# In[16]:


# Get stimulus epoch table for each session
epoch_table_A=data_set_A.get_stimulus_epoch_table()
epoch_table_B=data_set_B.get_stimulus_epoch_table()
epoch_table_C=data_set_C.get_stimulus_epoch_table()

# Make a list of epoch tables
epoch_tables={'epoch_table_A':epoch_table_A, 'epoch_table_B':epoch_table_B, 'epoch_table_C':epoch_table_C}

# Make lists of epochs within each session
epoch_A_list=epoch_table_A['stimulus'].unique()
epoch_B_list=epoch_table_B['stimulus'].unique()
epoch_C_list=epoch_table_C['stimulus'].unique()

# Make a list of epoch lists
epoch_lists=[epoch_A_list, epoch_B_list, epoch_C_list]

# Create a dictionary pairing epoch table names with their lists table
epoch_master={'epoch_table_A':epoch_A_list, 'epoch_table_B':epoch_B_list, 'epoch_table_C':epoch_C_list}
print epoch_master


# In[17]:


#Get deltaFF traces
Ca_A=A[1]
Ca_B=B[1]
Ca_C=C[1]
Ca_epoch_dict={'epoch_table_A':Ca_A, 'epoch_table_B':Ca_B, 'epoch_table_C':Ca_C}

# Find length of rows of np.array to find number of cells in sessino
cell_count=Ca_A.shape[0]


# In[18]:


# Function to create epoched delta f array traces for each cell over the 3 sessions in dictionary called "ca_trace_by_epoch"
# Like stimuli will be concatenated
def create_ca_arrays(epoch_tables, Ca_epoch_dict, epoch_master):
    ca_trace_by_epoch = {}
    for table_str in epoch_tables:
        Ca=Ca_epoch_dict[table_str]
        for stim_n in epoch_master[table_str]:
            curr_ca = []
            for ind, stim_row in epoch_tables[table_str].iterrows():
                if stim_row['stimulus'] == stim_n:
                    curr_ca.append(Ca[:, stim_row['start'] : stim_row['end']])
            curr_ca = np.concatenate(curr_ca, axis=1)
            ca_trace_by_epoch[stim_n] = curr_ca
    return ca_trace_by_epoch
            
# Run function to create epoched ca trace arrays
ca_trace_by_epoch=create_ca_arrays(epoch_tables, Ca_epoch_dict, epoch_master)
print(ca_trace_by_epoch)



# ## Use Data to Create Rolling Window Correlation Meauremtents with Minimum Threshold

# In[ ]:


# Determine the window size for rolling window of correlation
t_window = 100


for epoch,array in ca_trace_by_epoch.iteritems():
    trace_length=len(array[1]) # Length of calcium trace for cell in epoch
    cell_num=array.shape[0] # Get number of cells for exp container
    
    cor=[] #Create empty array to store correlation matrices
    
    for i in range(t_window, trace_length, t_window): #Loop through in bin sized steps
        x=array[:,(i-t_window):i] # Index through array in steps with rolling window
        cor.append(np.corrcoef(x)) # Take upper triangle of correlation matrix for each step and store in 'cor' array
        
    # Plot the histogram of correlation to see distribution of correlation across epoch
    plt.hist(cor[:]) 


# ## Get signal correlation

# In[85]:


for data in data_sets:
    # Import natural scenes
    ns={}
    from allensdk.brain_observatory.natural_scenes import NaturalScenes
    ns[data] = NaturalScenes(data)
print(ns)


# In[86]:


# Create list of session names
session_type={'session_id_A': 'three_session_A' , 'session_id_B': 'three_session_B', 'session_id_C': 'three_session_C'}

#Create analysis paths in dict called 'analysis path file'
analysis_path = os.path.join(drive_path,'ophys_experiment_analysis')
analysis_path_file={}
for session,idd in session_ids.iteritems():
    analysis_path_file[str(session)] = os.path.join(analysis_path, str(idd)+'_'+ str(session_type[session])+'_analysis.h5')
print analysis_path_file


# In[95]:


analysis_file=analysis_path_file['session_id_B']
f = h5py.File(analysis_file, 'r')
sc = f['analysis']['signal_corr_ns'].value
f.close()
#analysis_file = analysis_path_file['session_id_B']
#f = h5py.File(analysis_file, 'r')
#sc= f['analysis']['signal_corr_ns'].value
#sc= analysis_path_file['session_id_B']['analysis']['signal_corr_ns'].value
#analysis_file.close()


# In[98]:


# Print shape of one signal correlation to check structure
print sc.shape


# In[100]:


# Plot correlation matrix of cells for Natrual Scenes for three sessions
plt.imshow((sc), cmap='viridis', interpolation='none')
plt.xlabel("Cell #")
plt.ylabel("Cell #")
plt.title("Signal Correlations")
plt.colorbar()

