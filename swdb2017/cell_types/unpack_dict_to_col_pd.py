#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 12:46:43 2017

@author: annieciernia
"""

import numpy as np
import pandas as pd

#%%
def unpack(df, key_column, fillna=None):
    """
    Extracts contents of dictionary contained in Pandas dataframe to make new dataframe column
    Parameters
    ____________
    Input:
    df: dataframe with one column containing a dictionary of key:value pairs
    key_column: column with keys you want to extract into it's own column
    fillna: can be nan, or other desired value

    Output: dataframe with the selected key values from the dictionary as a new column.
    Concatinates onto original df
    
    Example:
    tf = pd.DataFrame([
        {'id': 1, 'nested': {'a': 1, 'b': 2} },
        {'id': 2, 'nested': {'a': 2, 'b': 4} },
        {'id': 3, 'nested': {'a': 3, 'b': 6} },
        {'id': 4, 'nested': {'a': 4}},
    ])
        
    unpack(tf, 'nested', 'nan')
                id  a  b
            0   1  1  2
            1   2  2  4
            2   3  3  6
            3   4  4  nan
    
    Modified from: https://codereview.stackexchange.com/questions/93923/extracting-contents-of-dictionary-contained-in-pandas-dataframe-to-make-new-data
    """
    

    ret = None
    if fillna is None:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[key_column].iteritems()))], axis=1)
        del ret[key_column]
    else:
        ret = pd.concat([df, pd.DataFrame((d for idx, d in df[key_column].iteritems())).fillna(fillna)], axis=1)
        del ret[key_column]
    return ret



#%%