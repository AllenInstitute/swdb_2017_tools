import pandas as pd

def has_pupil_data_df(boc,targeted_structures = None,stimuli = None,cre_lines = None):
    '''Returns full dataframe with only experiments that have pupil data for desired targeted structures, stimuli classes, 
    cre lines.
    
    Parameters
    ----------
            boc : BrainObservatoryCache from allensdk
            //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
            //boc = BrainObservatoryCache(manifest_file=manifest_file)
            
            targeted_structures : list
            brain region acronyms 
            
            stimuli : list 
            stimulus class names
            
            cre_lines : list 
            cre-line names
        
    Returns
    -------
            eye_df : dataframe 
            all experiments with eye-tracking data'''
    
       
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