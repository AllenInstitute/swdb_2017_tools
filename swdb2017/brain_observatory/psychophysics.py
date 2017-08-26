import pandas as pd


def get_natural_image_psychophysics_df():
    '''Returns dataframe with change detection psychophysics for 8 natural images. 
    
    Parameters
    ----------
        
    Returns
    -------
            beh_df : dataframe of behavioral response probabilities 8 natural images'''
    

    file_path = 'swdb_2017_tools/swdb2017/brain_observatory/resources/response_prob.csv'
    beh_df = pd.read_csv(file_path)
    
    return beh_df