import pandas as pd

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
    pupil_size : values of mouse pupil size. Can include NaNs
    timestamps : timestamps corresponding to pupil size values
    """
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=expt_session_id)
    pupil_size, timestamps = dataset.get_pupil_size()
    
    return pupil_size, timestamps

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
                pupil_size: pupil size values (list)
                time_stamps: time stamps corresponding to running speed values (list)"""
                    
    expt_session_ids = eye_df.id.values
    pupil_size_list = []
    for expt_id in expt_session_ids[:5]:
        timestamps, pupil_size = get_pupil_size_from_expt_session_id(boc, expt_id)
        pupil_size_list.append([expt_id, pupil_size, timestamps])
    pupil_df = pd.DataFrame(pupil_size_list, columns=['id', 'pupil_size', 'time_stamps'])
    
    return pupil_df