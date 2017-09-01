# Imports
import pandas as pd
from swdb2017.brain_observatory.behavior.get_pupil_size_data import has_pupil_data_df

def get_spont_corr_df(boc, tc, areas, cre_lines):
    """
    Create dataframe from saved data in trash_cache based on filtering criteria.

    Parameters:
    ----------
    boc : BrainObservatoryCache instance
    tc : TrashCache instance
    areas : list
        targeted structures (str), options: VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'
    cre_lines : list
        transgenic mouse lines (str), options : 'Cux2-CreERT2', 'Emx1-IRES-Cre', 'Nr5a1-Cre',
        'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre'

    Returns:
    -------
    spont_data_df : pandas dataframe
        contains metadata and peason correlation coefficients between behavioral features and df/f traces
        for spontaneous activity
    """

    feature_ids = ['pupil_area_rate', 'saccade_rate', 'pupil_area_smooth', 'running_speed_smooth',
                   'running_rate_smooth']

    # Get filtered dataframe
    exp_df = has_pupil_data_df(boc, targeted_structures=areas, stimuli=['spontaneous'], cre_lines=cre_lines)

    # Pull experiment data
    exp_idx = exp_df['id'].values
    experiments = tc.load_experiments(list(exp_idx))

    # Some other shit
    spont_data = []
    key = feature_ids[0]
    for exp in experiments:
        for c in exp['spont_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            spont_data.append(row)

    key = feature_ids[1]
    for exp in experiments:
        for c in exp['spont_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            spont_data.append(row)

    key = feature_ids[2]
    for exp in experiments:
        for c in exp['spont_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            spont_data.append(row)

    key = feature_ids[3]
    for exp in experiments:
        for c in exp['spont_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            spont_data.append(row)

    key = feature_ids[4]
    for exp in experiments:
        for c in exp['spont_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            spont_data.append(row)

    # Create dataframe
    spont_data_df = pd.DataFrame.from_records(spont_data)

    return spont_data_df


def get_ns_corr_df(boc, tc, areas, cre_lines):
    """
    Create dataframe from saved data in trash_cache based on filtering criteria.

    Parameters:
    ----------
    boc : BrainObservatoryCache instance
    tc : TrashCache instance
    areas : list
        targeted structures (str), options: VISal', 'VISam', 'VISl', 'VISp', 'VISpm', 'VISrl'
    cre_lines : list
        transgenic mouse lines (str), options : 'Cux2-CreERT2', 'Emx1-IRES-Cre', 'Nr5a1-Cre',
        'Rbp4-Cre_KL100', 'Rorb-IRES2-Cre', 'Scnn1a-Tg3-Cre'

    Returns:
    -------
    ns_data_df : pandas dataframe
        contains metadata and peason correlation coefficients between behavioral features and df/f traces
        for natural scenes
    """

    feature_ids = ['pupil_area_rate', 'saccade_rate', 'pupil_area_smooth', 'running_speed_smooth',
                   'running_rate_smooth']

    # Get filtered dataframe
    exp_df = has_pupil_data_df(boc, targeted_structures=areas, stimuli=['natural_scenes'], cre_lines=cre_lines)

    # Pull experiment data
    exp_idx = exp_df['id'].values
    experiments = tc.load_experiments(list(exp_idx))

    # Some other shit
    ns_data = []
    key = feature_ids[0]
    for exp in experiments:
        for c in exp['ns_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            ns_data.append(row)

    key = feature_ids[1]
    for exp in experiments:
        for c in exp['ns_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            ns_data.append(row)

    key = feature_ids[2]
    for exp in experiments:
        for c in exp['ns_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            ns_data.append(row)

    key = feature_ids[3]
    for exp in experiments:
        for c in exp['ns_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            ns_data.append(row)

    key = feature_ids[4]
    for exp in experiments:
        for c in exp['ns_behavior_correlation'][key].values:
            row = {'area': exp['metadata']['targeted_structure']}
            row['cre_line'] = exp['metadata']['cre_line']
            row['imaging_depth_um'] = exp['metadata']['imaging_depth_um']
            row['id'] = exp['metadata']['ophys_experiment_id']
            row[key] = c
            ns_data.append(row)

    # Create dataframe
    ns_data_df = pd.DataFrame.from_records(ns_data)

    return ns_data_df