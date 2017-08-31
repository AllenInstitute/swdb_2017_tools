import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding, MDS
from scipy.spatial.distance import squareform, pdist

def get_representational_similarity_NS(ns, corr='pearson', mean_sweep = True, which_trials = 'mean',
                                            embedding = None, n_components = 5, precision = None):
    """Returns the Representational Similarity Matrix for this session of natural scene reponse.
    Note: This should really be called the "Response Similarity Matrix" .
    
    Input
    -----
    ns = NaturalScenes object
    
    Options
    -------
    
    corr = 'spearman' or 'pearson'. Type of correlation to use.
    mean_sweep = True/False. Whether to use the mean sweep or, if false, the entire sweep
    which_trials =       How to handle multiple showings of the same image. Options:
                    'mean' : use trial-average mean response
                    'random' : use a random trial for each neuron. destroys effects noise correlation in the RSM
                    'all' : string all trials together 
    embedding = Whether to embed the responses before computing response similarity. Uses sklearn.
                Options: 'None', 'spectral_embedding', 'pca', 'mds'
    n_components = int >= 0. Number of components to use when embedding.
    
    """
    
    Nstim, N = ns.number_scenes, ns.numbercells
    N_trials = len(ns.stim_table)/Nstim

    
    if mean_sweep == True:
        if which_trials == 'mean':
            response = ns.response[:, :ns.numbercells, 0]
            
        elif which_trials == 'random':
            # first merge so we can select our choice of trial
            indexed_mean_sweep_response = pd.merge(ns.stim_table, ns.mean_sweep_response,
                     how='left', left_index=True, right_index =True)
            
            # awesome 1-liner to pick a random trial for each frame
            response = indexed_mean_sweep_response.groupby('frame').apply(
                        lambda x : x.iloc[np.random.randint(0,len(x))])
            
            # now drop the columns we picked up from stim_table. Also drop running response (last elem.)
            response = response.iloc[:,3:-1].values
            assert response.shape == (Nstim, N)
            
        elif which_trials == 'all':
        
            indexed_mean_sweep_response = pd.merge(ns.stim_table, ns.mean_sweep_response,
                     how='left', left_index=True, right_index =True)
            
            # the "response" to an frame will be the population reponse to each presentation of that frame,
            # concatenated
            response = np.zeros((Nstim, N * N_trials))
            for frame in range(Nstim):
                # get all the responses for a single frame. 
                this_frame = df.groupby('frame').get_group(frame-1)
                # drop stim_table columns and running resp.
                this_frame = this_frame.iloc[:,3:-1]
                
                response[frame] = this_frame.values.flatten()
            
        else:
            raise Exception('which_trials should be all, random, or mean.')
            
    elif mean_sweep == False:
        if which_trials == 'mean':
            avg_sweeps = get_avg_sweep(ns)
            # now unwrap
            response = unwrap_sweeps(avg_sweeps.iloc[:,:-1], ns)
            
        elif which_trials == 'random':
            rand_sweeps = get_rand_sweep(ns)
            # now unwrap
            response = unwrap_sweeps(ran_sweeps.iloc[:,:-1], ns)
        elif which_trials == 'all':
            
            indexed_mean_sweep_response = pd.merge(ns.stim_table, ns.mean_sweep_response,
                     how='left', left_index=True, right_index =True)
            # this super-gross operation returns a pd.Series of length number_scenes,
                # with each value being the concatenated all-trial, all-neuron, all-sweep response
            my_group = indexed_sweep_response.groupby('frame').apply(
                        lambda x: np.concatenate(
                                        [np.concatenate(
                                                x[str(neuron)].values) for neuron in range(N)]  ))
            
            # and then we turn it into an array
            response = np.array([r for r in my_group ])

            
        else:
            raise Exception('which_trials should be all, random, or mean.')   
    else:
        raise Exception('mean_sweep must be True or False')
        
    rep_sim = get_representational_similarity_response(response, corr=corr, precision = precision,
                                            embedding = embedding, n_components = n_components)

    return rep_sim

def get_representational_similarity_response(response, corr='pearson',  embedding = None, n_components = 5,precision = None,
                                                        prefit_embedder = None):
    """Returns the Representational Similarity Matrix for this response matrix (numpy array).
    Note: This should really be called the "Response Similarity Matrix" .
    
    Input
    -----
    ns = NaturalScenes object
    
    Options
    -------
    
    corr = 'spearman' or 'pearson' or 'mahalanobis'. Type of correlation or distance metric to use.
    mean_sweep = True/False. Whether to use the mean sweep or, if false, the entire sweep
    which_trials =       How to handle multiple showings of the same image. Options:
                    'mean' : use trial-average mean response
                    'random' : use a random trial for each neuron. destroys effects noise correlation in the RSM
                    'all' : string all trials together 
    embedding = Whether to embed the responses before computing response similarity. Uses sklearn.
                Options: 'None', 'spectral_embedding', 'pca', 'mds'
    n_components = int >= 0. Number of components to use when embedding.
    
    """
    
    
        
    if embedding is not None:
        if prefit_embedder is None:
            if embedding in ['pca', 'PCA']:
                embed = PCA(whiten = True, n_components=n_components)

            elif embedding == 'spectral_embedding':
                embed = SpectralEmbedding(n_components=n_components, n_jobs = -1)

            elif embedding in ['mds', 'MDS']:
                embed = MDS(n_components=n_components)
            else:
                raise Exception('embedding must be None, pca, mds, or spectral_embedding')
            response = embed.fit_transform(response)
        else:
            prefit_embedder.transform(response)


    Nstim = response.shape[0]
    rep_sim = np.zeros((Nstim, Nstim))
    rep_sim_p = np.empty((Nstim, Nstim))
    if corr == 'pearson':
        rep_sim = np.corrcoef(response)
        rep_sim = np.triu(rep_sim) + np.triu(rep_sim, 1).T # fill in lower triangle


    elif corr == 'spearman':
        for i in range(Nstim):
            for j in range(i, Nstim): # matrix is symmetric
                rep_sim[i, j], _ = st.spearmanr(response[i], response[j])
        rep_sim = np.triu(rep_sim) + np.triu(rep_sim, 1).T # fill in lower triangle
        
        
    elif corr == 'mahalanobis':
        if precision is None:
            #using sklearn for its nice approximate inverse methods
            if response.shape[0] < response.shape[1]:
                raise('Must have more stimuli responses than neurons')

            emp_cov = EmpiricalCovariance()
            emp_cov.fit(response)
            precision = emp_cov.get_precision()
            
        for i in range(Nstim):
            for j in range(i, Nstim): # matrix is symmetric
                delta = response[i]-response[j]
                rep_sim[i, j] = np.dot(np.dot(delta, precision), delta)
                
        rep_sim = np.triu(rep_sim) + np.triu(rep_sim, 1).T # fill in lower triangle
        
    else:
        # Assume other sklearn distance metric
        rep_sim = squareform(pdist(response, metric=corr))

    # set diagonal to zero for plotting
    np.fill_diagonal(rep_sim,0)

    return rep_sim


def get_avg_sweep(ns):
    """Takes a NaturalScenes object and returns a dataframe of shape (numberscenes, numbercells)
    containing the sweep averaged over trials.
    """
    indexed_sweep_response = pd.merge(ns.stim_table, ns.sweep_response,
                     how='left', left_index=True, right_index =True)

    # awesome 1-liner to grab the average sweep for each frame
    response = indexed_sweep_response.groupby('frame').apply(
                              lambda x :x.mean())
    return response.iloc[:,3:]


def unwrap_sweeps(sweep_df, ns):
    """Takes a dataframe containing a sweep for each neuron and returns an 'unwrapped' numpy array,
    of shape (numberscenes, numbercells * N_trials)"""
    Nstim, N = ns.number_scenes, ns.numbercells
    sweep_len = len(sweep_df.iloc[0,0])
    
    unwrapped = np.zeros((Nstim, N * sweep_len))
    for frame in range(Nstim):
        # get all the responses for a single frame
        this_frame = np.hstack(sweep_df.iloc[frame])
        unwrapped[frame] = this_frame
        
    return unwrapped
    
def get_rand_sweep(ns):
    """Takes a NaturalScenes object and returns a dataframe of shape (numberscenes, numbercells)
    containing, for each neuron and frame, a random sweep from one of the trials.
    """
    indexed_sweep_response = pd.merge(ns.stim_table, ns.sweep_response,
                     how='left', left_index=True, right_index =True)

    # awesome 1-liner to grab a random sweep for each frame
    response = indexed_sweep_response.groupby('frame').apply(
                              lambda x :x.iloc[np.random.randint(0,len(x))])
    return response.iloc[:,3:]
