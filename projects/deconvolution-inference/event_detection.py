from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import sklearn.mixture as mixture
from skimage.restoration import denoise_tv_chambolle
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
import pickle, os

def get_events_mixture(dff, dfftime, plot=False):

    '''
    this doesn't work very well
    :param dff:
    :param dfftime:
    :param plot:
    :return:
    '''
    N, T = dff.shape
    dpgmm = mixture.GMM(n_components=2, covariance_type='diag', n_iter=1000, tol=1e-3)
    event_train = np.zeros((N, T))

    for i in range(N):

        dff_i = dff[i].reshape(-1, 1)

        dpgmm.fit(dff_i)
        Y = dpgmm.predict(dff_i)

        components = np.unique(Y)
        means = dpgmm.means_
        noise_component = components[np.where(means == np.amin(means))[0]]

        Y[Y == noise_component] = 0.
        Y[Y != 0] = 1.

        Ydiff = np.diff(Y)
        Ydiff2 = Y[2:] - Y[:-2]
        Ydiff3 = Y[3:] - Y[:-3]
        Ydiff = np.pad(Ydiff, pad_width=(1, 0), mode='constant')
        Ydiff2 = np.pad(Ydiff2, pad_width=(1, 1), mode='constant')
        # Ydiff3 = np.pad(Ydiff3, pad_width=(1, 2), mode='constant')

        events = (Ydiff == 1) * (Ydiff2 == 1) #* (Ydiff3 == 1)

        event_train[i, events] = 1


    if plot:
        plt.figure();
        plt.plot(dfftime, 7*event_train[i], 'b')
        plt.plot(dfftime, dff_i, 'k')
        plt.plot(dfftime[Y==0], dff_i[Y==0], 'ko')
        plt.plot(dfftime[Y==1], dff_i[Y==1], 'ro')

    return event_train


def get_events_derivative(dff_trace, k_min=0, k_max=10, delta=3, smooth_window=5, smooth_weight=0.3, plot=False):
    '''
    this seems to work ok
    :param dff_trace:
    :param k_min:
    :param k_max:
    :param delta:
    :param first_only:
    :param smooth_window:
    :param smooth_weight:
    :param plot:
    :return:
    '''
  
    dff_trace = smooth(dff_trace, smooth_window)
    # if smooth_weight > 0:
    #     dff_trace = denoise_tv_chambolle(dff_trace, weight=smooth_weight)
    var_dict = {}

    for ii in range(len(dff_trace)):

        if ii + k_min >= 0 and ii + k_max <= len(dff_trace):
            trace = dff_trace[ii + k_min:ii + k_max]

            xx = (trace - trace[0])[delta] - (trace - trace[0])[0]
            # yy = (trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2]
            yy = max((trace - trace[0])[delta + 2] - (trace - trace[0])[0 + 2],
                     (trace - trace[0])[delta + 3] - (trace - trace[0])[0 + 3],
                     (trace - trace[0])[delta + 4] - (trace - trace[0])[0 + 4])

            var_dict[ii] = (trace[0], trace[-1], xx, yy)

    xx_list, yy_list = [], []
    for _, _, xx, yy in var_dict.itervalues():
        xx_list.append(xx)
        yy_list.append(yy)

    mu_x = np.median(xx_list)
    mu_y = np.median(yy_list)

    xx_centered = np.array(xx_list) - mu_x
    yy_centered = np.array(yy_list) - mu_y

    std_factor = 1
    std_x = 1. / std_factor * np.percentile(np.abs(xx_centered), [100 * (1 - 2 * (1 - sps.norm.cdf(std_factor)))])
    std_y = 1. / std_factor * np.percentile(np.abs(yy_centered), [100 * (1 - 2 * (1 - sps.norm.cdf(std_factor)))])

    curr_inds = []
    allowed_sigma = 4
    for ii, (xi, yi) in enumerate(zip(xx_centered, yy_centered)):
        if np.sqrt(((xi) / std_x) ** 2 + ((yi) / std_y) ** 2) < allowed_sigma:
            curr_inds.append(True)
        else:
            curr_inds.append(False)

    curr_inds = np.array(curr_inds)
    data_x = xx_centered[curr_inds]
    data_y = yy_centered[curr_inds]
    Cov = np.cov(data_x, data_y)
    Cov_Factor = np.linalg.cholesky(Cov)
    Cov_Factor_Inv = np.linalg.inv(Cov_Factor)

    # ===================================================================================================================

    # fig_dff, ax_dff = plt.subplots()
    # ax_dff.plot(dff_trace, 'k')

    # fig, ax = plt.subplots()
    noise_threshold = max(allowed_sigma * std_x + mu_x, allowed_sigma * std_y + mu_y)
    mu_array = np.array([mu_x, mu_y])
    yes_list, no_list, size_list = [], [], []

    for ii, (t0, tf, xx, yy) in var_dict.iteritems():

        xi_z, yi_z = Cov_Factor_Inv.dot((np.array([xx, yy]) - mu_array))

        # # Conditions in order:
        # # 1) Outside noise blob
        # # 2) Minimum change in df/f
        # # 3) Change evoked by this trial, not previous
        # # 4) At end of trace, ended up outside of noise floor
        #
        # if np.sqrt(xi_z ** 2 + yi_z ** 2) > 4 and yy > .05 and xx < yy and tf > noise_threshold / 2:

        # Conditions in order:
        # 1) outside noise blob
        # 2) positive transient
        # 3) change evoked by this trial, not next

        if np.sqrt(xi_z**2 + yi_z**2) > 4 and xx > 0:
        # if np.sqrt(xi_z ** 2 + yi_z ** 2) > 4 and yy > .05 and xx < yy and tf > noise_threshold / 2:

            yes_list.append(ii)
            size_list.append(xx)

            # ax.plot([xx], [yy], 'b.')
            # ax_dff.plot(ii, 2., 'b')
        else:
            no_list.append(ii)
            # ax.plot([xx], [yy], 'r.')

    # events_temp[yes_list] = 1

    if plot:
        plt.figure()
        plt.plot(xx_list[yes_list], yy[yes_list], 'b.')
        plt.plot(xx_list[no_list], yy[no_list], 'r.')

    yes_array = np.array(yes_list)
    size_array = np.array(size_list)

    return yes_array, size_array


def concatenate_adjacent_events(times, heights, delta=3):

    '''
    for each sequence of events in adjacent frames:
     if fewer than delta frames, take the max height
     if greater than delta frames, take the max in windows of length delta and then add

    :param times:
    :param heights:
    :param delta:
    :return:
    '''

    time_diff = np.diff(times)
    time_diff = np.pad(time_diff, pad_width=(1, 0), mode='constant')

    time_diff[0] += 10
    time_ind = np.where(time_diff > 1)[0]
    times_new = times[time_ind]

    time_diff[0] -= 20
    time_end = np.where(time_diff > 1)[0]

    Ntimes = len(times_new)
    heights_new = np.zeros(Ntimes)

    for i in range(Ntimes-1):
        if time_ind[i+1] - time_ind[i] <= delta:
            heights_new[i] = np.amax(heights[time_ind[i]:time_ind[i + 1]])
        else:
            heights_temp = []
            for j in range(time_ind[i], time_ind[i+1], delta):
                heights_temp.append(np.amax(heights[j:j+delta]))

            heights_new[i] = sum(heights_temp)

        # spkHeights_new[i] = dff[spkTimes[spk_ind[i+1]-1]+delta] - dff[spkTimes[spk_ind[i]]+delta]

    heights_new[Ntimes-1] = np.amax(heights[time_ind[Ntimes-1]:])

    return times_new, heights_new


def smooth(x, window_len=11, window='hanning', mode='valid'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode=mode)
    return y


def get_event_triggered_dff(dff, spkTimes, spkHeights, width=100, plot=False):

    ### trigger on spiketimes
    Nevents = len(spkTimes)
    if len(dff.shape) > 1:
        dff = dff[0]
    T = dff.shape

    event_triggered_dff = []
    # width_peak = 5

    plt.figure()
    for i in range(Nevents):
        if spkTimes[i]+width < T:
            event = dff[spkTimes[i]:spkTimes[i]+width]

            peak_ind = np.where(event == np.amax(event))[0]
            event_triggered_dff.append(event / event[peak_ind])
            plt.plot(range(-peak_ind, width-peak_ind), event, 'k')

            # try:
            #     # event_triggered_dff.append(dff[spkTimes[i]:spkTimes[i]+width])
            #     # event_triggered_dff[i] -= event_triggered_dff[i][0]
            #     # event_triggered_dff[i] = (event_triggered_dff[i] - dff[spkTimes[i]]) / (np.amax(event_triggered_dff[i][:width_peak]) - dff[spkTimes[i]])
            #
            # except:
            #     continue

    event_triggered_dff = np.array(event_triggered_dff)

    mean_event = np.mean(event_triggered_dff, axis=0)
    # mean_event -= np.mean(event[-width/10:])
    # mean_event = mean_event[:-1]
    # mean_event = np.pad(mean_event, pad_width=(1, 0), mode='constant')

    mean_event -= mean_event[0]
    ind0 = np.where(mean_event < 0)[0]
    if len(ind0) > 0:
        mean_event = mean_event[:ind0[0]]

    if plot:
        plt.figure()
        plt.plot(event_triggered_dff.T, 'k')
        plt.plot(mean_event, 'r', linewidth=2)

    return mean_event/np.amax(mean_event)


def get_biexponential_filter(width=100, tau1=20., tau2=2.):

    if tau1 < tau2:
        tau1_new = tau2
        tau2_new = tau1
        tau1 = tau1_new*1
        tau2 = tau2_new*1

    x = np.arange(0, width)
    filt = np.exp(-x/tau1) - np.exp(-x/tau2)
    return filt / np.amax(filt)


def get_events_integral(dff, smooth_weight=0.3, outlier=True):

    '''
    lost my old iteration of "dff goes up" event detection and this version doesn't work as well
    :param dff:
    :param smooth_weight:
    :param outlier:
    :return:
    '''

    if smooth_weight > 0:
        dff = denoise_tv_chambolle(dff, weight=smooth_weight)
    thresh = 0.03

    d_dff = np.diff(dff)
    d_dff = np.hstack((d_dff, -1.))
    times = np.where(d_dff > thresh)[0]

    d_times = np.diff(times)
    times2 = times[np.where(d_times == 1)]
    times = times[np.where(d_times > 1)] # the ones where it doesn't increase on consecutive bins

    times2 = np.hstack((-10, times2))
    d_times2 = np.diff(times2)
    ind = np.where(d_times2 > 1)[0]

    for i in ind:
        times = np.hstack((times, times2[1+i]))

    heights = []
    for i, t in enumerate(times):
        ind = np.where(d_dff[t:] < 0)[0]

        if len(ind) > 0:
            heights.append(dff[t+ind[0]] - dff[t])

        else:
            continue

    heights = np.array(heights)

    if outlier:
        ind = np.where(heights > np.median(heights) / 0.6745)[0]

        times = times[ind]
        heights = heights[ind]

    plt.figure(); plt.plot(dff); plt.plot(times, heights, 'ko', markersize=2)

    return times, heights


def example():

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    ecs = boc.get_experiment_containers(simple=False)

    # ecs = ecs[:1]

    ec_ids = [ec['id'] for ec in ecs]
    exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids, simple=False)
    expt_list = [exp['id'] for exp in exps]

    expt = expt_list[0]
    data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)

    dffTime, dff = data_set.get_dff_traces()
    N, T = dff.shape

    ### do for one cell
    times, heights = get_events_derivative(dff[0], smooth_window=5, plot=False, smooth_weight=0.2)
    train = np.zeros((T))
    train[times] = heights

    times_new, heights_new = concatenate_adjacent_events(times, heights)
    train_new = np.zeros((T))
    train_new[times_new] = heights_new

    plt.figure()
    plt.plot(smooth(dff[0], 5), 'k')
    # plt.plot(dff[0], 'k')
    # plt.plot(dff[0], 'k', linewidth=1, alpha=0.8)
    # plt.plot(denoise_tv_chambolle(dff[0], weight=0.2), 'k')
    plt.plot(train, 'b')
    plt.plot(train_new, 'r')


def get_events_all_bob_experiments():

    save_dir = '/local1/Documents/projects/cam_analysis/events'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
    ecs = boc.get_experiment_containers(simple=False)

    ec_ids = [ec['id'] for ec in ecs]
    exps = boc.get_ophys_experiments(experiment_container_ids=ec_ids)
    expt_list = [exp['id'] for exp in exps]

    bad_expt_list = [563226901]
    expt_list = [exp for exp in expt_list if exp not in bad_expt_list]

    for exp_num, expt in enumerate(expt_list):  # [:1]:

        print 'experiment #'+str(exp_num)+'/'+str(len(expt_list))
        data_set = boc.get_ophys_experiment_data(ophys_experiment_id=expt)

        dffTime, dff = data_set.get_dff_traces()
        N, T = dff.shape

        events = [[] for i in range(N)]

        for i in range(N):
            if np.mod(i, 50) == 0:
                print 'roi #'+str(i)+'/'+str(N)
            times, heights = get_events_derivative(dff[i], smooth_window=5, plot=False, smooth_weight=0.2)
            events[i] = concatenate_adjacent_events(times, heights)

        savefile = os.path.join(save_dir, str(expt) + '.pkl')
        pickle.dump(events, open(savefile, 'wb'))


if __name__ == '__main__':
    get_events_all_bob_experiments()
