def get_experiment_ids(boc,targeted_structure,cre_line,imaging_depth):
    '''Gives back session IDs for an experiment
    specified by targeted structure, cre-line, and imaging depth

    Parameters
    ----

    targeted_structure: list
    cre_line: list
    imaging_depth: list

    '''
    exps = boc.get_experiment_containers(targeted_structures=targeted_structure,
                                       cre_lines=cre_line,imaging_depths=imaging_depth)
    ids = [exp['id'] for exp in exps]
    return ids

def get_rsms(boc, drive_path, targeted_structures,cre_lines,imaging_depths,stimuli):
    ''' Gets response similarity matrices for all areas, cre-lines, and imaging depths specified

        Parameters
        ----------
        boc : object
        stimuli : string
            'drifting_gratings', 'locally_sparse_noise', 'locally_sparse_noise_4deg',
            'locally_sparse_noise_8deg', 'natural_movie_one', 'natural_movie_three', 'natural_movie_two',
            'natural_scenes', 'spontaneous', or 'static_gratings'
        drive_path : string
            path to the Brain Observatory Cache
        ids: list
            List of experiment container ids.
        targeted_structures: list
            List of structure acronyms.  Must be in the list returned by
            BrainObservatoryCache.get_all_targeted_structures().
        imaging_depths: list
            List of imaging depths.  Must be in the list returned by
            BrainObservatoryCache.get_all_imaging_depths().
        cre_lines: list
            List of cre lines.  Must be in the list returned by
            BrainObservatoryCache.get_all_cre_lines().'''
    rsms = []
    experiment_ids=get_experiment_ids(boc,targeted_structures,cre_lines,imaging_depths) #self.experiment_ids
    exps = pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=experiment_ids, stimuli=stimuli)).copy()
    for experiment_id in experiment_ids:
        exp = pd.DataFrame(boc.get_ophys_experiments(experiment_container_ids=[experiment_id], stimuli=stimuli))
        session_id = exp['id'][0]
        data_set = boc.get_ophys_experiment_data(ophys_experiment_id=session_id)
        if stimuli[0] == 'natural_scenes':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_B_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_ns'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0] == 'static_gratings':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_A_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_sg'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0] == 'drifting_gratings':
            analysis_path = os.path.join(drive_path, 'ophys_experiment_analysis')
            analysis_file = os.path.join(analysis_path, str(session_id) + '_three_session_A_analysis.h5')
            f = h5py.File(analysis_file, 'r')
            rs = f['analysis']['rep_similarity_dg'].value
            rsms.append(rs)
            f.close()
        elif stimuli[0]=='natural_movies':
            #for later
            print 'sorry not yet'
        else:
            raise Exception('stimulus not found')
    exps['rsm'] = rsms
    return exps


