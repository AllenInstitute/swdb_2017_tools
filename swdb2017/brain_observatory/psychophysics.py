import pandas as pd
import pkg_resources



def get_natural_image_psychophysics_df():
    '''Returns dataframe with change detection psychophysics for 8 natural images. 
    
    Parameters
    ----------
        
    Returns
    -------
            beh_df : dataframe of behavioral response probabilities 8 natural images'''
    

  
    file_path = pkg_resources.resource_filename('response_prob.csv', 'swdb2017.brain_observatory.resources')
    beh_df = pd.read_csv(file_path)
    
    return beh_df