#Get Brain Observatory Experimental Data Functions 
   
import numpy as np
import os
import h5py

def get_all_representational_similarity_matrices(BrainObservatoryCache, session_type, 
                    stimulus_type, drive_path, file_name=None, ids=None, 
                    targeted_structures=None, imaging_depths=None, 
                    cre_lines=None, transgenic_lines=None, 
                    include_failed=False):
    '''
        Parameters
        ----------
        BrainObservatoryCache : object
        session_type: string
            'three_session_A', 'three_session_B', or 'three_session_C'
        stimulus_type : string
            'drifting_gratings', 'locally_sparse_noise', 'locally_sparse_noise_4deg', 
            'locally_sparse_noise_8deg', 'natural_movie_one', 'natural_movie_three', 'natural_movie_two', 
            'natural_scenes', 'spontaneous', or 'static_gratings'
        drive_path : string
            path to the Brain Observatory Cache
        file_name: string
            File name to save/read the experiment containers.  If file_name is None,
            the file_name will be pulled out of the manifest.  If caching
            is disabled, no file will be saved. Default is None.
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
            BrainObservatoryCache.get_all_cre_lines().
        transgenic_lines: list
            List of transgenic lines. Must be in the list returned by
            BrainObservatoryCache.get_all_cre_lines() or.
            BrainObservatoryCache.get_all_reporter_lines().
        include_failed: boolean
            Whether or not to include failed experiment containers.

        Returns
        -------
        rs : a 3-d array of representational similarity matrices from your drive_path for the specified
        session and stimulus type. The 3-d array is a concatenation of all 2-d representational
        similarity matrices for the desired experiments (experiment is the 3rd dimension.  

        
        Notes
        -----
        stimulus_type must be found within the given session_type. See the white paper documentation
        on the Allen Brain Institute website to determine what session_type contains each stimulus_type
    '''
    
    print('Loading representational similarity matrices')
    
    #load all experiments of interest
    exps = BrainObservatoryCache.get_experiment_containers(file_name=file_name, ids=ids, 
                                         targeted_structures=targeted_structures, 
                                         imaging_depths=imaging_depths, 
                                         cre_lines=cre_lines, 
                                         transgenic_lines=transgenic_lines, 
                                         include_failed=include_failed)
    
    #Determine the size of the images of interest
    first_session_id = BrainObservatoryCache.get_ophys_experiments(
                experiment_container_ids=[exps[0]['id']], 
                session_types = [session_type])[0]['id']
    image_set = BrainObservatoryCache.get_ophys_experiment_data(ophys_experiment_id=first_session_id) 
    images = image_set.get_stimulus_template(stimulus_type)    
    n_stim = images.shape[0]+1
    
    #Loop through the experiments and add the representational similarity matrix to rs
    rs = np.zeros([n_stim,n_stim,len(exps)])
    for exp in range(len(exps)):
        
        if exp%20 == 0:
            print('working on '+ str(exp) + ' of ' + str(len(exps)))
        
        session_id = BrainObservatoryCache.get_ophys_experiments(
                experiment_container_ids=[exps[exp]['id']], 
                session_types = [session_type])[0]['id']
        data_set = BrainObservatoryCache.get_ophys_experiment_data(ophys_experiment_id=session_id) 
        
        # This is where we access the representational similarity matrix
        from allensdk.brain_observatory.natural_scenes import NaturalScenes
        ns = NaturalScenes(data_set)
        analysis_path = os.path.join(drive_path,'ophys_experiment_analysis')
        analysis_file = os.path.join(analysis_path, str(session_id)+ '_' + str(session_type) + '_analysis.h5')
        f = h5py.File(analysis_file, 'r')
        rs[:,:,exp] = f['analysis']['rep_similarity_ns'].value
        f.close()
        
    return rs

