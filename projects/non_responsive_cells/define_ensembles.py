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
    ctrlspksbinar = np.zeros((num_cells,time.shape[0]))
    time_resamp = np.arange(T[0],T[1],binsize)+binsize/2

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
            rndStrt = T[0] + np.floor(rand * ntime_free / dt) * dt    #randomly chosen start time, quantised to original spike-time resolution
            rndStrt = rndStrt.reshape(1,1)
            shufTs = np.hstack([rndStrt, rndStrt+np.cumsum(IEI[:,ind])]) # starting from randomly chosen start time, the times of new events in shuffled train 1,
            if np.max(shufTs) > T[1]:
                # print 'last event in shuffled train occurs too late'  # paranoia - should rarely happen, except with pathological cases (too few spikes)
                nattempts += 1
                if nattempts > 20:
                    print 'Cannot satisfactorily shuffle this spike-train'
                    break
            else:
                break

        # binarize
        idx = find_closest(time[:],shufTs[0,:])
        ctrlspksbinar[cellID,idx] = 1  # add event indices, add to shuffled onset times vector

    ctrlhist = np.sum(ctrlspksbinar,axis=0)

    # pad with zeros and reshape for binning
    ctrlhist_binned = make_bin(ctrlhist, time, time_resamp)

    return time_resamp, ctrlhist_binned



def find_high_activity(spktms, T, dt, binsize, nsurr=1000, pval = 0.05):
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
        ensembles_w_cells: numpy array
            2D array [cells, ensembles] giving the binarized periods of "high activity", as compared with surrogate

        time_w_ensembles: numpy array
            1D array of the time ensembles occur

    Notes
    ---------
        Currently doesn't output cell IDs for this ensemble. Am working on it.
        Alternatively can output times of ensemble if that's easier, but just gave
        binarized output as per Jacob's request...

    '''
    cutoff = nsurr*(1-pval)

    # binarize original data set
    time, spktms_binar = binarize_spktms(spktms,T,dt)

    # find coactive periods
    coactive = spktms_binar.sum(axis=0)

    # bin original data set
    time_resamp = np.arange(T[0],T[1],binsize)+binsize/2

    # pad coactive with zeros to divide into bins, and reshape
    coactive_binned = make_bin(coactive, time, time_resamp)

    surpasses_surr = np.zeros(coactive_binned.shape)

    # find cont of binarized spktms in larger bins (pad with zeros to divide into bins, and reshape)
    spktms_binned = np.zeros([spktms.shape[0],coactive_binned.shape[0]])
    for i in range(spktms_binned.shape[0]):
        spktms_binned[i] = make_bin(spktms_binar[i,:], time, time_resamp)

    is_spk_binned =  np.array(spktms_binned>0,dtype=int)

    # for every surrogate data set
    for _ in range(nsurr):
        # shuffle and bin network activity
        time_resamp, coactive_surr_binned = shuffle_intervals(spktms,T,dt,binsize)
        #compare surrogate with original data and count where exceeds
        surpasses_surr += np.array(coactive_binned>coactive_surr_binned,dtype=int)

    ensembles = np.array(surpasses_surr > cutoff, dtype=int)
    ensembles_w_cells = is_spk_binned[:,ensembles==1]
    time_w_ensembles = time_resamp[ensembles==1]
    return time_w_ensembles, ensembles_w_cells


def make_bin(x, t, t_new):
    # pad with zeros
    num_zeros_to_pad = t_new.shape[0]-(t.shape[0]%t_new.shape[0])
    x_new = np.zeros(x.shape[0]+num_zeros_to_pad)
    x_new[:x.shape[0]] = x

    x_new = x_new.reshape( t_new.shape[0], x_new.shape[0]/t_new.shape[0])
    x_new = np.sum(x_new, axis=1)

    return x_new


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


def get_ensemble_info(info, recording_type_str):

    i = 0
    maxspks = 0
    cellID = np.empty(len(info[1][recording_type_str].keys()))
    time = info[0][recording_type_str]

    for key, value in info[1][recording_type_str].iteritems():
        maxspks = np.maximum(maxspks,len(value))

    spktms=np.empty([cellID.shape[0],maxspks])
    spktms[:] = np.nan

    for key, value in info[1][recording_type_str].iteritems():
        cellID[i] = key
        spktms[i,:len(value)] = value
        i += 1

    return time, spktms, cellID
