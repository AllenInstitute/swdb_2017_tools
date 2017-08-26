

def bandpass_filter(frames, low_cuttoff, high_cutoff, frame_rate, order=4, rp=0.1):
    ''' Band pass filter a signal on its time axis.

    This function performs a chebyshev-1 filter
    The filter is done on the time axis of the signal
    and is used to remove potential artifacts on the signal.
    For example bleaching of the fluorophors (very low frequency)
    as well as high frequency noise from the CCD, CMOS, PMT, etc

    input:
        frames: The signal to be filtered. Ensure it has the following shape:
                    (time, space). I've used it with (number_frames, width, height)
        low_cuttoff: The low frequency cuttoff of the filter
        high_cutoff: The high frequency cuttoff of the filter
        frame_rate: The sampling frequency of your acquisition
        order: The order of the polynomial that is used to calculate the filter.
                    by default this is 4
        rp: The maximum ripple allowed. Units are in decibels

    output:
        frames: The filtered signal, with original shape

    
    Further reading and sources:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.signal.cheby1.html
        https://en.wikipedia.org/wiki/Chebyshev_filter#Type_I_Chebyshev_filters
    '''
    from scipy import signal

    # Calculate the bandlimit of the signal
    nyq = frame_rate/2.0
    low_cuttoff= low_cuttoff / nyq
    high_cutoff = high_cutoff / nyq
    wn = [low_cuttoff, high_cutoff]
    
    b, a = signal.cheby1(order, rp, wn, 'bandpass', analog=False)
    frames = signal.filtfilt(b, a, frames, axis=0)

    return frames



if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    arr = np.sin(np.linspace(0,100,1000))
    noisy_arr = arr + np.sin(np.linspace(0, 500, 1000))**2 + 10*np.exp(-np.linspace(0, 5, 1000))
    filtered_arr = bandpass_filter(noisy_arr, 1, 2, 100)
    func_arr = [arr, noisy_arr, filtered_arr]
    title_arr = ["Noiseless Signal", "High frequency artifact.", "Recovered Signal"]
    colors = ['r', 'g', 'b']

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    for i in range(3):
        ax[i].plot(func_arr[i], colors[i])
        ax[i].set_title(title_arr[i])
        ax[i].set_ylabel("Signal Amplitude")
        ax[i].set_xlabel("Frames (100Hz)")
    fig.tight_layout()
    plt.show()