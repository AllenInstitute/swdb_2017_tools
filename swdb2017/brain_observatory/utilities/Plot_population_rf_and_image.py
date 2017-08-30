# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:55:13 2017

@author: saskiad
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import h5py
import allensdk.brain_observatory.stimulus_info as si
from allensdk.brain_observatory.observatory_plots import plot_mask_outline

def plot_scene_and_experiment_pop_rf(boc, experiment_id, natural_scene_id):
    '''Retrieves, aligns, and plots a natural scene with a population receptive field overlain

Parameters
----------
first : brain observatory cache
second : experiment id
third: natural scene id


Returns
-------
natural scene template (unaligned)
population receptive field subunit(unaligned)

    '''
    ns_template = get_natural_scene_template_expt(boc,experiment_id)
    pop_rf = get_population_rf(boc, experiment_id)
    plot_scene_with_pop_rf(pop_rf, ns_template, natural_scene_id)
    return (ns_template, pop_rf)

def plot_movie_and_experiment_pop_rf(boc, experiment_id, natural_movie_id, frame_number):
    '''Retrieves, aligns, and plots a natural movie frame with a population receptive field overlain

Parameters
----------
first : brain observatory cache
second : experiment id
third: natural movie id, string ('one','two','three')
fourth: natural movie frame number


Returns
-------
natural movie template (unaligned)
population receptive field subunit(unaligned)

    '''
    nm_template = get_natural_movie_template_expt(boc, experiment_id, natural_movie_id)
    pop_rf = get_population_rf(boc, experiment_id)
    plot_movie_with_pop_rf(pop_rf, nm_template, frame_number)
    return (nm_template, pop_rf)
    
def get_natural_scene_template_expt(boc, experiment_id):
    ns_session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], 
                                              stimuli=['natural_scenes'])[0]['id']
    data_set_ns = boc.get_ophys_experiment_data(ophys_experiment_id=ns_session_id)
    ns_template = data_set_ns.get_stimulus_template('natural_scenes')
    return ns_template

def get_natural_movie_template_expt(boc, experiment_id, natural_movie_id):
    natural_movie_name = 'natural_movie_'+natural_movie_id
    nm_session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], 
                                              stimuli=[natural_movie_name])[0]['id']
    data_set_nm = boc.get_ophys_experiment_data(ophys_experiment_id=nm_session_id)
    nm_template = data_set_nm.get_stimulus_template(natural_movie_name)
    return nm_template

def get_population_rf(boc, experiment_id):
    c_flag = 'C'
    lsn_name = 'locally_sparse_noise'
    rf_name = 'receptive_field_lsn'
    #
    for a in boc.get_ophys_experiments(experiment_container_ids=[experiment_id]):
        if a['session_type'].endswith('2'):
            c_flag = 'C2'
            if a['targeted_structure'] != 'VISp':
                lsn_name = 'locally_sparse_noise_8deg'
                rf_name = 'receptive_field_lsn8'
            else:
                lsn_name = 'locally_sparse_noise_4deg'
                rf_name = 'receptive_field_lsn4'

    drive_path = boc.manifest.get_path('BASEDIR')
    if c_flag=='C':
        session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C_analysis.h5')
    elif c_flag=='C2':    
        session_id = boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=[lsn_name])[0]['id']
        analysis_file = os.path.join(drive_path, 'ophys_experiment_analysis', str(session_id)+'_three_session_C2_analysis.h5')

    
    f = h5py.File(analysis_file, 'r')
    receptive_field = f['analysis'][rf_name].value
    f.close()
    pop_rf = np.nansum(receptive_field, axis=(2,3))
    return pop_rf

def plot_scene_with_pop_rf(pop_rf, image_template, image_number):
    '''Aligns, and plots a natural scene with a population receptive field overlain

Parameters
----------
first : population receptive field
second : natural scene template
third: natural scene id

    '''
    m = si.BrainObservatoryMonitor()
    pop_rf[pop_rf>0]=255
    rf_convert = m.lsn_image_to_screen(pop_rf, origin='upper')
    rf_convert[np.where(rf_convert<200)]=0
    image = image_template[image_number,:,:]
    plt.figure()
    ax = plt.subplot(111)
    m = si.BrainObservatoryMonitor()
    m.show_image(m.natural_scene_image_to_screen(image, origin='upper'),ax=ax, show=False, origin='upper')
    plot_mask_outline(rf_convert, ax, color='b')
    plt.show()

def plot_movie_with_pop_rf(pop_rf, nm_template, frame_number):
    '''Aligns, and plots a natural movie frame with a population receptive field overlain

Parameters
----------
first : population receptive field
second : natural movie template
third: natural movie frame number

    '''
    m = si.BrainObservatoryMonitor()
    pop_rf[pop_rf>0]=255
    rf_convert = m.lsn_image_to_screen(pop_rf, origin='upper')
    rf_convert[np.where(rf_convert<200)]=0
    image = nm_template[frame_number,:,:]
    plt.figure()
    ax = plt.subplot(111)
    m = si.BrainObservatoryMonitor()
    m.show_image(m.natural_movie_image_to_screen(image, origin='upper'),ax=ax, show=False, origin='upper')
    plot_mask_outline(rf_convert, ax, color='b')
    plt.show()

