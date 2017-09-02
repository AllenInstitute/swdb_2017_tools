import numpy as np
import pandas as pd

def calc_mod(msr_f,threshold,mod_col='dx'):
    """Calculate modulation index for mean sweep response

    Parameters
    ----------
    msr_f : mean_sweep_response table for a specific frame
        Allen Institute experiment file

    threshold : float
        cut point for behavior metric

    mod_col : str

    Returns
    -------

    run_mod : np.ndarray

    """
    stat = msr_f[
        msr_f[mod_col]<threshold
    ]
    run = msr_f[
        msr_f[mod_col]>threshold
    ]
    stat = stat.mean().drop(mod_col).values
    run = run.mean().drop(mod_col).values
    run_mod = []
    i=0
    for rn,st in zip(run,stat):
        if rn > st:
            rmax = rn
            rmin = st
            c = 1
        else:
            rmax = st
            rmin = rn
            c = -1
        mod_i = c*((rmax-rmin)/np.abs(rmax))
        run_mod.append(mod_i)
        i+=1
    return run_mod
