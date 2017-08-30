##################################################################
# define_ensembles.py
#   Contains functions to identify periods of high activity (ensembles) 
#   by generating and comparing with shuffled surrogate data
#   kferguson
##################################################################
import numpy as np
import pandas as pd


def binarize_spktms(spktms,T,dt):
    ''' Returns time and binarized spiketimes
    
    Parameters
    -----------
    spkts: numpy array
            2D np array: [cells, spike times], where spike times are seconds
    
        T: numpy array
            T = [Tstart, Tend], the start and end times (respectively) in seconds
            for the recording/event from which spkts was recorded.
            
        dt: float
            the quantisation step of the original spike times (in seconds)
            
    Returns
    ---------
    time: numpy array
        1D array of you time based on T, dt, in seconds

    spktms_bin: numpy array
        2D array: [cell, binarized spike times]
    
    '''
    
    time = np.arange(T[0],T[1],dt)
    spktms_bin = np.zeros([spktms.shape[0],time.shape[0]])
    
    for cellID in range(spktms.shape[0]):
        spktms_finite = spktms[cellID, np.isfinite(spktms[cellID,:])]
        idx = find_closest(time,spktms_finite)
        spktms_bin[cellID,idx] = 1
        
    return time, spktms_bin



def shuffle_intervals(spkts,T,dt,binsize): 
    ''' Returns the ID arrays for time and for the sum of shuffled ISIs (binarized) across the network
    
    Parameters
    ----------
        spkts: numpy array
            2D np array: [cells, spike times], where spike times are seconds
    
        T: numpy array
            T = [Tstart, Tend], the start and end times (respectively) in seconds
            for the recording/event from which spkts was recorded.
            
        dt: float
            the quantisation step of the original spike-times (in seconds)
        
        binsize: float
            the bin width in seconds used for summing and comparing network activity
            
    Returns
    ---------
        time_resamp: numpy array
            1D array giving time points for larger bins if binsize>dt.  If binsize=dt, is 
            just your original time points based on T, dt.  Time in seconds.
              
        ctrlhist: numpy array
            1D array giving the binarized sum of network activity based on shuffled ISIs.
    
    Notes
    ----------
    Possible that the function could get stuck in an infinite loop in
    pathological cases; so returns with error message after 20 attempts
    
    Shuffling ISI part is based on MATLAB code from Mark Humphries 6/10/2012
    '''

    num_cells = spkts.shape[0]
    time = np.arange(T[0],T[1],dt)
    time_resamp = np.arange(T[0],T[1],binsize)
    ctrlspksbinned = np.zeros((num_cells,time_resamp.shape[0]))

    for cellID in range(num_cells): 
        Train = spkts[cellID,:]  # onset times in seconds
        Train = Train[np.isfinite(Train)].reshape(1,len(Train[np.isfinite(Train)]))
        IEI = np.diff(Train)  # inter-event intervals for train
        IStrt = Train[:,0]-T[0]
        IEnd = T[1] - Train[:,-1]   # time before first event; time after last
        ntime_free = IStrt + IEnd                      # total amount of free time before first and after last event
        ind = np.arange(IEI.shape[1])
        nattempts = 0
        while True:      
            np.random.shuffle(ind)            # randomly shuffle indices of intervals in train
            rand =  np.random.random((1,))
            rndStrt = T[0] + np.ceil(rand * ntime_free / dt) * dt    #randomly chosen start time, quantised to original spike-time resolution
            rndStrt = rndStrt.reshape(1,1)
            shufTs = np.hstack([rndStrt, rndStrt+np.cumsum(IEI[:,ind])]) # starting from randomly chosen start time, the times of new events in shuffled train 1,
            if np.max(shufTs) > T[1]:
                print 'last event in shuffled train occurs too late'  # paranoia - should rarely happen, except with pathological cases (too few spikes)
                nattempts += 1
                if nattempts > 20:
                    print 'Cannot satisfactorily shuffle this spike-train'
                    break
            else:
                break
  
        # binarize
        idx = find_closest(time_resamp[:],shufTs[0,:])
        ctrlspksbinned[cellID,idx] = 1  # add event indices, add to shuffled onset times vector
    
    ctrlhist = np.sum(ctrlspksbinned,axis=0)
    return time_resamp, ctrlhist


def find_high_activity(spktms, nsurr=1000, pval = 0.05, dt=1./30, binsize=0.250, T=[0,3550]):
    '''Returns binarized periods of "high activity", as compared with surrogate
        
    Parameters
    ----------
        spkts: numpy array
            2D np array: [cells, spike times], where spike times are seconds
        
        nsurr: int
            the number of surrogate datasets you want to compare with
    
        pval: float
            the p-value you want to surpass when comparing with surrogate datasets.
        
        dt: float
            the quantisation step of the original spike-times (in seconds)
 
        binsize: float
            the bin width in seconds used for summing and comparing network activity
 
        T: numpy array
            T = [Tstart, Tend], the start and end times (respectively) in seconds
            for the recording/event from which spkts was recorded.           
        
            
    Returns
    ---------
        ensembles: numpy array
            1D array giving the binarized periods of "high activity", as compared with surrogate
        
        time: numpy array
            1D array of you time based on T, dt, in seconds
 
    Notes
    --------- 
        Currently doesn't output cell IDs for this ensemble. Am working on it.  
        Alternatively can output times of ensemble if that's easier, but just gave
        binarized output as per Jacob's request...            
        
    '''
    cutoff = nsurr*(1-pval)
    time, spktms_bin = binarize_spktms(spktms,T,dt)  
       
    coactive = spktms_bin.sum(axis=0)
    surpasses_surr = np.zeros(coactive.shape)
    coactive_surr_tot = np.zeros(coactive.shape)
       
    for _ in range(nsurr):
        time_resamp, coactive_surr = shuffle_intervals(spktms,T,dt,dt)
        coactive_surr_tot += coactive_surr
        surpasses_surr += np.array(coactive>coactive_surr,dtype=int)
    
    coactive_surr_tot = coactive_surr_tot/nsurr
    
    ensembles = np.array(surpasses_surr > cutoff, dtype=int)
    return ensembles, time



def find_closest(A, target):
    '''
    finds the closest indices between an array and it's target. 
    '''

    #A must be sorted
    idx = A.searchsorted(target)
    idx = np.clip(idx, 1, len(A)-1)
    left = A[idx-1]
    right = A[idx]
    idx -= target - left < right - target
    return idx
    

