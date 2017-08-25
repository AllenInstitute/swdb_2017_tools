def has_pupil_data(boc,targeted_structures = None,stims = None):
    '''Returns full dataframe with only experiments that have pupil data.
    Input parameters:
            boc = BrainObservatoryCache from allensdk
            //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
            //boc = BrainObservatoryCache(manifest_file=manifest_file)
            
            structures = list of brain region acronyms, strings (optional)
            
            stims = list of stimulus class names, strings (optional)
            
    Output parameters:
            eye_df = dataframe containing all experiments with eye-tracking data'''
    
    #Imports
    import pandas as pd
       
    if targeted_structures is None:
        targeted_structures = boc.get_all_targeted_structures()
    if stims is None:
        stims = boc.get_all_stimuli()     
   
    filtered_df = pd.DataFrame(boc.get_ophys_experiments(stimuli=stims,targeted_structures = targeted_structures,simple = False))

    eye_df = filtered_df[(filtered_df['fail_eye_tracking'] == False)]
    
    return eye_df
    