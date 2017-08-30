import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter

def has_pupil_data_df(boc,targeted_structures = None,stimuli = None,cre_lines = None):
    '''Returns full dataframe with only experiments that have pupil data for desired targeted structures, stimuli classes, 
    cre lines.
    
    Parameters
    ----------
    boc : BrainObservatory Cache from allensdk
    //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
    //boc = BrainObservatoryCache(manifest_file=manifest_file)
    
    targeted_structures : list of brain region acronyms 
    
    stimuli : list of stimulus class names
    
    cre_lines : list of cre-line names
        
    Returns
    -------
    eye_df : dataframe of experiments with eye-tracking data'''
    
       
    if targeted_structures is None:
        targeted_structures = boc.get_all_targeted_structures()
    if stimuli is None:
        stimuli = boc.get_all_stimuli()  
    if cre_lines is None:
        cre_lines = boc.get_all_cre_lines()
   
    filtered_df = pd.DataFrame(boc.get_ophys_experiments(stimuli=stimuli,
                                                         targeted_structures = targeted_structures,
                                                         cre_lines = cre_lines,
                                                         simple = False))

    eye_df = filtered_df[(filtered_df['fail_eye_tracking'] == False)]
    
    return eye_df    

def get_pupil_size_from_expt_session_id(boc, expt_session_id, remove_outliers=True):
    """
    Author: @marinag
    Adapted by: @blaje42
    """
    """
    Get pupil size for a single experiment session.
    Parameters
    ----------
    boc: Brain Observatory Cache instance
    expt_session_id : ophys experiment session ID

    Returns
    -------
    pupil_size : values of mouse pupil size (area, pixels^2). Can include NaNs
    timestamps : timestamps corresponding to pupil size values
    """
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=expt_session_id)
    pupil_size, timestamps = dataset.get_pupil_size()
    
    return pupil_size, timestamps

def convert_pupil_area_to_diameter(data):
    '''Convert pupil size data (which is area of the pupil in pixels squared) to diameter (in pixels)
    Parameters
    ----------
    data : array containing pupil area trace (pixels^2)
    
    Returns
    -------
    pupil_diameter : array containing pupil diameter trace (pixels) '''
    
    pupil_diameter = np.sqrt(data/np.pi)
    return pupil_diameter

def get_pupil_df(boc, eye_df):
    """Returns dataframe containing behavior data from Allen Brain Observatory.
    Parameters
    ----------
    boc : BrainObservatory Cache from allensdk
    //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
    //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
    //boc = BrainObservatoryCache(manifest_file=manifest_file)
    
    targeted_structures : list of brain region acronyms 
    
    stimuli : list of stimulus class names
    
    cre_lines : list of cre-line names
    
    Returns
    -------
    pupil_df : dataframe containing:
        id: unique experiment id (int)
        pupil_area_pixels2: pupil size values (area, pixels^2) (array)
        pupil_diameter_pixels: pupil diameter values (pixels) (array)
        time_stamps: time stamps corresponding to running speed values (array)"""
                    
    expt_session_ids = eye_df.id.values
    pupil_size_list = []
    for expt_id in expt_session_ids:
        timestamps, pupil_size = get_pupil_size_from_expt_session_id(boc, expt_id)
        pupil_diameter = convert_pupil_area_to_diameter(pupil_size)
        pupil_size_list.append([expt_id, pupil_size, pupil_diameter, timestamps])
    pupil_df = pd.DataFrame(pupil_size_list, columns=['id', 'pupil_size_pixels2','pupil_diameter_pixels', 'time_stamps'])
    
    return pupil_df

def remove_nans(pupil_df,keys):
    '''Remove NaN values from specified key columns and also same indices from time_stamps. Will remove the same set of 
    indices for all columns based on first key in keys list.
    
    Parameters
    ----------
    pupil_df : dataframe containing data and time_stamps column
    keys : list of keys from which you want NaNs removed (NOT including time_stamps, this is automatic)
    
    Returns
    -------
    pupil_df_nonans : dataframe same as pupil_df with NaNs removed and corresponding time_stamp indices removed'''
    
    pupil_df_nonans = pupil_df.copy()
    keys.append('time_stamps')
    
    #Keep only indices with non-nan data points
    for j in range(len(pupil_df_nonans[keys[0]])):
        idx_nonans = ~np.isnan(pupil_df_nonans[keys[0]][j])
        for key in keys:            
            pupil_df_nonans[key][j] = pupil_df_nonans[key][j][idx_nonans]

    return pupil_df_nonans     

def smooth_data(pupil_df_nonans,keys,sigma):
    '''Gaussian smooth data in desired columns specified by keys
    
    Parameters
    ----------
    pupil_df : dataframe containing data
    keys : list of strings of columns containing data to be smoothed
    sigma : int, sigma value to input to gaussian filter
    
    Returns
    -------
    pupil_df_smooth : dataframe with original data plus additional columns containing smoothed data
    '''
    
    new_keys = []
    for i,key in enumerate(keys):
        new_keys.append(key + '_smooth')
        
    pupil_df_smooth = pupil_df_nonans.copy()
    
    #Insert empty columns for smooth data
    for i,key in enumerate(new_keys):
        df_keys = list(pupil_df_smooth.keys())
        idx = df_keys.index(keys[i])
        pupil_df_smooth.insert(idx+1,column = key,value = "")
   
    #Smooth data and add to dataframe
    for j in range(len(pupil_df_nonans[keys[0]])):
        for i,key in enumerate(new_keys):
            y = gaussian_filter(pupil_df_nonans[keys[i]][j],sigma)
            pupil_df_smooth[key][j] = y
            
    return pupil_df_smooth

