import pandas as pd


def get_expt_session_ids_from_expt_container_ids(boc, expt_container_ids, stimuli=None):
    """Collect list of experiment session IDs for selected experiment container IDs

    Parameters
    ----------
    boc: Brain Observatory Cache instance
    expt_container_ids : experiment container IDs from boc.get_experiment_containers()
    stimuli: list of stimulus names to restrict returned experiment session IDs to only sessions that contain that stimulus type
        stimuli = None returns all experiment session IDs for the selected experiment containers

    Returns
    -------
    expt_session_ids : list of experiment session IDs
    """
    expt_session_ids = []
    for expt_container_id in expt_container_ids:
        expt_session_info = boc.get_ophys_experiments(experiment_container_ids=[expt_container_id], stimuli=stimuli)
        expt_session_df = pd.DataFrame(expt_session_info)
        for session_id in expt_session_df.id.values:
            expt_session_ids.append(session_id)
    return expt_session_ids


