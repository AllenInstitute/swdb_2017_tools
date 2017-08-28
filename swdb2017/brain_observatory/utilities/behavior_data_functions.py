import os
import pandas as pd
import numpy as np

def get_filtered_df(boc, targeted_structures=None, stims=None, cre_lines=None):
    """Returns dataframe filtered by stimulus inputs and targeted structure inputs from Brain Observatory data.
    Input parameters:
            boc = BrainObservatoryCache from allensdk
            //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
            //boc = BrainObservatoryCache(manifest_file=manifest_file)

            targeted_structures = list of brain region acronyms, strings (optional)

            stims = list of stimulus class names, strings (optional)

            cre_lines = list of cre line names, strings (optional)

    Output parameters:
            filtered_df = dataframe containing experiments only with stim or targeted_structures inputs"""

    if targeted_structures is None:
        targeted_structures = boc.get_all_targeted_structures()

    if stims is None:
        stims = boc.get_all_stimuli()

    if cre_lines is None:
        cre_lines = boc.get_all_cre_lines()

    filtered_df = pd.DataFrame(
        boc.get_ophys_experiments(stimuli=stims, targeted_structures=targeted_structures, cre_lines=cre_lines, simple=False))

    return filtered_df

def get_running_speed_from_expt_session_id(boc, expt_session_id, remove_outliers=True):
    """
    Author: @marinag
    """
    """
    Get running speed trace for a single experiment session.
    Parameters
    ----------
    boc: Brain Observatory Cache instance
    expt_session_id : ophys experiment session ID
    remove_outliers : Boolean. If True, replace running trace outlier values (>100, -10) with NaNs
    Returns
    -------
    running_speed : values of mouse running speed in centimeters per second. Can include NaNs
    timestamps : timestamps corresponding to running speed values
    """
    dataset = boc.get_ophys_experiment_data(ophys_experiment_id=expt_session_id)
    running_speed, timestamps = dataset.get_running_speed()
    if remove_outliers:
        running_speed = remove_running_speed_outliers(running_speed)
    return running_speed, timestamps


def get_behavior_df(boc, df):
    """Returns dataframe containing behavior data from Allen Brain Observatory.
        Input parameters:
                boc = BrainObservatoryCache from allensdk
                //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
                //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
                //boc = BrainObservatoryCache(manifest_file=manifest_file)

                df = dataframe containing experiment ids for desired behavioral data output


        Output parameters:
                behavior_df = dataframe containing:
                    id: unique experiment id (int)
                    speed_cm_s: running speed values in cm/s; can include NaNs (list)
                     time_stamps: time stamps corresponding to running speed values (list)"""

    expt_session_ids = df.id.values

    running_speed_list = []
    for expt_id in expt_session_ids[:5]:
        running_speed, timestamps = get_running_speed_from_expt_session_id(boc, expt_id, remove_outliers=False)
        running_speed_list.append([expt_id, running_speed, timestamps])

    behavior_df = pd.DataFrame(running_speed_list, columns=['id', 'speed_cm_s', 'time_stamps'])

    return behavior_df


def plot_running_speed(boc, behavior_df):
    """Plot all running speed traces for a set of ophys experiments within a pandas dataframe containing behavior data
    -------
    Input parameters:
            boc = BrainObservatoryCache from allensdk
            //from allensdk.core.brain_observatory_cache import BrainObservatoryCache
            //manifest_file = os.path.join(drive_path,'brain_observatory_manifest.json')
            //boc = BrainObservatoryCache(manifest_file=manifest_file)

            behavior_df: dataframe containing id (int), speed_cm_s (list), and time_stamps (list)

    Output parameters:
            ax: axes handle

    """
    x_values = behavior_df['time_stamps']
    y_values = behavior_df['speed_cm_s']
    expt_ids = behavior_df['id']

    for i in range(len(expt_ids)):
        plt.figure()
        plt.plot(x_values[i], y_values[i])
        plt.title('Experiment ID: ' + str(expt_ids[i]))
        plt.xlabel('Time (s)')
        plt.ylabel('Running Speed (cm/s)')