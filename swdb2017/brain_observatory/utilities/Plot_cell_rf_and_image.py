# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:25:26 2017

@author: saskiad
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import allensdk.brain_observatory.stimulus_info as si
from allensdk.brain_observatory.observatory_plots import plot_mask_outline

def plot_scene_and_cell_rf(boc, cell_specimen_id, natural_scene_id):
    '''Retrieves, aligns, and plots a natural scene with a cell's receptive field overlain

Parameters
----------
first : brain observatory cache
second : cell specimen id
third: natural scene id


Returns
-------
natural scene template (unaligned)
receptive field ON subunit(unaligned)
receptive field OFF subunit(unaligned)

    '''
    ns_template = get_natural_scene_template(boc, cell_specimen_id)
    rf_on, rf_off = get_rf_mask(boc, cell_specimen_id)
    plot_scene_with_rf(rf_on, rf_off, ns_template, natural_scene_id)
    return (ns_template, rf_on, rf_off)

def plot_movie_and_cell_rf(boc, cell_specimen_id, natural_movie_id, frame_number):
    '''Retrieves, aligns, and plots a natural movie frame with a cell's receptive field overlain

Parameters
----------
first : brain observatory cache
second : cell specimen id
third: natural movie id (string: 'one','two','three')
forth: natural movie frame


Returns
-------
natural ovie template (unaligned)
receptive field ON subunit(unaligned)
receptive field OFF subunit(unaligned)

    '''
    nm_template = get_natural_movie_template(boc, cell_specimen_id, natural_movie_id)
    rf_on, rf_off = get_rf_mask(boc, cell_specimen_id)
    plot_movie_with_rf(rf_on, rf_off, nm_template, frame_number)
    return (nm_template, rf_on, rf_off)

def get_natural_scene_template(boc, cell_specimen_id):
    ns_session_id = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id], 
                                              stimuli=['natural_scenes'])[0]['id']
    data_set_ns = boc.get_ophys_experiment_data(ophys_experiment_id=ns_session_id)
    ns_template = data_set_ns.get_stimulus_template('natural_scenes')
    return ns_template

def get_natural_movie_template(boc, cell_specimen_id, natural_movie_id):
    natural_movie_name = 'natural_movie_'+natural_movie_id
    nm_session_id = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id], 
                                              stimuli=[natural_movie_name])[0]['id']
    data_set_nm = boc.get_ophys_experiment_data(ophys_experiment_id=nm_session_id)
    nm_template = data_set_nm.get_stimulus_template(natural_movie_name)
    return nm_template

def get_rf_mask(boc, cell_specimen_id):
    c_flag = 'C'
    lsn_name = 'locally_sparse_noise'
    for a in boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id],
                                       stimuli=['three_session_C', 'three_session_C2']):
        if a['session_type'].endswith('2'):
            c_flag = 'C2'
            if a['targeted_structure'] != 'VISp':
                lsn_name = 'locally_sparse_noise_8deg'
            else:
                lsn_name = 'locally_sparse_noise_4deg'

    drive_path = boc.manifest.get_path('BASEDIR')
    
    if c_flag=='C':
        session_id = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C_analysis.h5')
    elif c_flag=='C2':    
        session_id = boc.get_ophys_experiments(cell_specimen_ids=[cell_specimen_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C2_analysis.h5')
    
    data_set = boc.get_ophys_experiment_data(session_id)
    if cell_specimen_id not in data_set.get_cell_specimen_ids():
        raise Exception("cell %d not in experiment session %d" % (cell_specimen_id, session_id))
    cell_index = data_set.get_cell_specimen_indices(cell_specimen_ids=[cell_specimen_id])[0]
    f = h5py.File(analysis_file, 'r')
    rf_on = f['analysis'][lsn_name][str(cell_index)]['on']['fdr_mask']['data'].value
    rf_off = f['analysis'][lsn_name][str(cell_index)]['off']['fdr_mask']['data'].value
    f.close()    
    return(rf_on[0,:,:], rf_off[0,:,:])

def plot_scene_with_rf(rf_on, rf_off, image_template, image_number):
    '''Aligns and plots natural scene with cell receptive field

Parameters
----------
first : receptive field on subnits
second : receptive field off subunit
third: natural scene template
forth: natural scene id
    '''
    m = si.BrainObservatoryMonitor()
    on_convert = m.lsn_image_to_screen(rf_on, origin='upper')
    on_convert[np.where(on_convert<250)]=0
    off_convert = m.lsn_image_to_screen(rf_off, origin='upper')
    off_convert[np.where(off_convert<250)]=0
    image = image_template[image_number,:,:]
    plt.figure()
    ax = plt.subplot(111)
    m = si.BrainObservatoryMonitor()
    m.show_image(m.natural_scene_image_to_screen(image, origin='upper'),ax=ax, show=False, origin='upper')
    plot_mask_outline(off_convert, ax, color='b')
    plot_mask_outline(on_convert, ax, color='r')
    plt.show()

def plot_movie_with_rf(rf_on, rf_off, nm_template, frame_number):
    '''Aligns and plots natural movie frame with cell receptive field

Parameters
----------
first : receptive field on subnits
second : receptive field off subunit
third: natural movie template
forth: natural movie frame number
    '''
    m = si.BrainObservatoryMonitor()
    on_convert = m.lsn_image_to_screen(rf_on, origin='upper')
    on_convert[np.where(on_convert<250)]=0
    off_convert = m.lsn_image_to_screen(rf_off, origin='upper')
    off_convert[np.where(off_convert<250)]=0
    image = nm_template[frame_number,:,:]
    plt.figure()
    ax = plt.subplot(111)
    m = si.BrainObservatoryMonitor()
    m.show_image(m.natural_movie_image_to_screen(image, origin='upper'),ax=ax, show=False, origin='upper')
    plot_mask_outline(off_convert, ax, color='b')
    plot_mask_outline(on_convert, ax, color='r')
    plt.show()
    
    
    
    
