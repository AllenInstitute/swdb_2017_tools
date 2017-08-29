import pandas as pd
import numpy as np
def get_run_mod_cells(boc,stimuli='natural_scenes',p_max=1):
    """Get all run modulated cells and optionally set a max p-value to filter by.

        Parameters
        ----------
        boc : BrainObservatoryCache
        stimuli : str
            Currently limited to static_gratings,drifting_gratings, and natural_scenes
        p_max : int
            Max p-value; only return records with p-value<p_max
        Returns
        -------
        out : pandas.DataFrame
    """

    cell_df = pd.DataFrame(boc.get_cell_specimens())
    run_mod_cols = dict(
        static_gratings=['p_run_mod_sg','run_mod_sg'],
        drifting_gratings=['p_run_mod_dg','run_mod_dg'],
        natural_scenes=['p_run_mod_ns','run_mod_ns']
    )

    run_cells = cell_df[
        (cell_df[run_mod_cols[stimuli][1]].notnull())
        & (cell_df[run_mod_cols[stimuli][0]]<p_max)
    ]
    return run_cells
