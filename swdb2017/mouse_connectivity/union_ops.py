import numpy as np

def get_structure_unionizes(mcc,injection_structure_ids,target_structure_ids):
    '''Get connectivity unionize records for injections in certain structures for a subset of summary structures

    Parameters
    ----------
    mcc : MouseConnectivityCache
        instantiated mouse connectivity cache.

    injection_structure_ids : 1D np.ndarray
        array of injection_structure_ids

    target_structure_ids : 1D np.ndarray
        array of target_structure_ids

    Returns
    -------
    union records : pandas.core.frame.DataFrame
        DataFrame of union records originating from source structure ids that hit target_structure_ids.

    '''

    exps = mcc.get_experiments(injection_structure_ids=injection_structure_ids)
    exp_ids = [exp['id'] for exp in exps]
    unions = mcc.get_structure_unionizes(experiment_ids=exp_ids,structure_ids=target_structure_ids)

    return unions

def get_summary_structure_unionizes(mcc,injection_structure_ids):
    summary_structures = structure_tree.get_structures_by_set_id([167587189])
    summary_stucture_ids = [s['id'] for s in summary_structures]
    return get_structure_unionizes(mcc,injection_structure_ids,summary_structure_ids)
