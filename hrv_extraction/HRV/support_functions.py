"""
Last updated on Thur 29/12/2022

@author: Ismoil; PanopticAI


"""

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.fft import fft, fftfreq
from pyampd.ampd import find_peaks
from kymatio.numpy import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory

def peak_detection_gt(signal, window_size=20, fs=60, pyamd_win=100):
    
    '''
    This function cleans IBIs of the ground truth PPG signal.
    Input:
        signal (N,) = (array) signal of size N
        window_size = (float) size of the windowing in units of number of peaks
        fs = (int) framerate
        pyamd_win = (int) size of the windowing for AMPD algorithm.
        {Colak, A. M., Shibata, Y., & Kurokawa, F. (2016, November).
         FPGA implementation of the automatic multiscale based peak detection for real-time signal analysis on renewable energy systems.
         In 2016 IEEE International Conference on Renewable Energy Research and Applications (ICRERA) (pp. 379-384). IEEE.}

    Output:
        final_ibi (M,) = (list) IBIs in units of ms
        
    '''
    
    peak_locations = find_peaks(signal, pyamd_win)
    ibi = np.diff(peak_locations)*1000/fs
    
    L = len(ibi)
    final_ibi = []
    for w in np.arange(0, L, window_size):
        if w + window_size >= len(ibi):
            ibi_window = ibi[w:]

        else:
            ibi_window = ibi[w: w + window_size]

        mean_ibi_window = np.mean(ibi_window)

        min_val_window = mean_ibi_window*0.7
        max_val_window = mean_ibi_window*1.3
        
        
        ibi_window = ibi_window[ibi_window > min_val_window]
        ibi_window = ibi_window[ibi_window < max_val_window]

        final_ibi = final_ibi + ibi_window.tolist()
    
    return np.array(final_ibi)


def peak_analysis(signal, window_size=10, fs=60, pyamd_win=100):
    '''
    This function calculates instantenious IBIs from given rPPG/PPG/ECG signal.

    Input:
        signal (N,) = (array) signal of size N
        window_size = (float) size of the windowing in units of number of peaks
        fs = (int) framerate
        pyamd_win = (int) size of the windowing for AMPD algorithm.
        {Colak, A. M., Shibata, Y., & Kurokawa, F. (2016, November).
         FPGA implementation of the automatic multiscale based peak detection for real-time signal analysis on renewable energy systems.
         In 2016 IEEE International Conference on Renewable Energy Research and Applications (ICRERA) (pp. 379-384). IEEE.}

    Output:
        final_ibi (M,) = (list) IBIs in units of ms
    '''
    peak_locations = find_peaks(signal, pyamd_win)
    ibi = np.diff(peak_locations)*1000/fs
    # reject physically impossible IBIs
    ibi = ibi[ibi < 1300]
    ibi = ibi[ibi > 400]

    # reject impossible jumps in IBIs
    mean_ibi = np.mean(ibi)
    min_val = mean_ibi*0.4
    max_val = mean_ibi*1.6

    L = len(ibi)
    final_ibi = []
    for w in np.arange(0, L, window_size):
        if w + window_size >= len(ibi):
            ibi_window = ibi[w:]

        else:
            ibi_window = ibi[w: w + window_size]
        
        ibi_window = ibi_window[ibi_window > min_val]
        ibi_window = ibi_window[ibi_window < max_val]
        mean_ibi_window = np.mean(ibi_window)

        min_val_window = mean_ibi_window*0.8
        max_val_window = mean_ibi_window*1.2
        

        ibi_window = ibi_window[ibi_window > min_val_window]
        ibi_window = ibi_window[ibi_window < max_val_window]

        final_ibi = final_ibi + ibi_window.tolist()

    return final_ibi

def bandpass_filter(signal, order, fs, min_freq=0.8, max_freq=4.5, apply_hanning=False, need_details=False):
    '''This function applies bandpass filter and returns filtered signal. Reccommended values are useful for Heart Rate.
    Sosfiltfilt function of SciPy locolizes the position of the peaks because it has zero phaze.

    THIS FUNCTION CANNOT BE USED IN REAL TIME


    Input: 
        signal (N,) = (array) signal of size N
        order = (int) order of butterworth signal
        fs = (int) framerate
        min_freq = (float) min frequnecy range for HR in units of Hz
        max_freq = (float) max frequnecy range for HR in units of Hz
        apply_hanning = (bool) whether to apply hanning windowing
        need_details = (bool) whether to return details from FFT 

    Output:
        y (N,) = (array) bandpass filtered signal
        fft_details = (dictionary) FFT details such as FFT freqs, amplitudes and maximum freq in FFT range (only returned if need_details = True)

    '''

    # Step 1: IIR Bandpass filter
    sos = butter(order, [min_freq, max_freq],
                 btype='bandpass', output='sos', fs=fs)
    result = sosfiltfilt(sos, np.double(signal))
    y = result

    if apply_hanning:
        hanning = np.hanning(len(result))
        y = hanning * result

    fft_details = {}
    if need_details:

        N = len(signal)
        T = 1 / fs

        yf = fft(y)

        freqs = fftfreq(N, T)[:N//2]  # get real valuesof freqs

        yf = 2.0/N * np.abs(yf[0:N//2])  # get real parts of amplitudes

        max_idx = np.where(yf == np.amax(yf))
        highest_freq = float(freqs[max_idx])

        fft_details['frequencies'] = freqs
        fft_details['amplitudes'] = yf
        fft_details['max_freq'] = highest_freq

        return y, fft_details

    return y


def inter_cleaning(signal, fs=30, band=0.4, Use_HeartPY=False):
    '''
    This function is designed to clean rPPG signals. Can be used for other signals
    by adjusting parameters. Recommended values are for HR normal ranges

    Input:
        signal (N,) = (array) signal of size N
        fs = (int) framerate
        band  = (float) freq band for butterworth bandpass filtering
        Use_HeartPY = (bool) whether to use HeartPY peak detection algorithm.
                        Recommended to use for relatively clean signals. 

    Output:
        finalPPG (N,) = (array) cleaned signal
    '''



    ppg1, mydict = bandpass_filter(
        signal, 5, fs, 0.8, 2.5, apply_hanning=True, need_details=True)
    highest_freq = mydict['max_freq']

    # This part is still under development which deals with second peaks. Subsequent works need to be done.
    

    mean_ibi = 1/highest_freq
    lower_band = 1/(mean_ibi*(1.1 + band))
    upper_band = 1/(mean_ibi*(1 - band))

    if lower_band < 0.7:
        ppg3 = bandpass_filter(ppg1, 4, fs, 0.7, upper_band)
    elif upper_band > 2.5:
        ppg3 = bandpass_filter(ppg1, 4, fs, lower_band, 2.5)
    else:
        ppg3 = bandpass_filter(ppg1, 4, fs, lower_band, upper_band)
    
    #ppg2 = bandpass_filter(signal, 4, fs, 2*lower_band, 2*upper_band)
    finalPPG = ppg3  # + ppg2
    return finalPPG



def inter_cleaning1_2(signal, fs=64, band = 0.3, compression = False):
    '''
    This function is designed to clean rPPG signals. It is part of the pipeline 
    for HRV1.2. Uses ScatteringTransform as basis
    
    Input:
        signal (N,) = (array) signal of size N
        fs = (int) framerate
        band  = (float) freq band for butterworth bandpass filtering

    Output:
        finalPPG (N,) = (array) cleaned signal
    '''
    
    if fs == 60 or fs == 64:
        J = 10
    elif fs == 28 or  fs == 30 or fs == 35 or fs == 32:
        J = 9
    else:
        J = 8
        
    Q = 2*J  
    T = len(signal)
    
    _, psi1_f, _, _ = scattering_filter_factory(12, J, Q)
    XI = []
    SIGMA = []
    for psi in psi1_f:
        XI.append(psi['xi'])
        SIGMA.append(psi['sigma'])
    XI = np.array(XI)*fs
    
    scattering = Scattering1D(J, T, Q)
    meta = scattering.meta()
    order1 = np.where(meta['order'] == 1)
    Sx = scattering(signal)
    Sx1 = Sx[order1]
    

    val1 = np.argmax(XI < 2.5) 
    val2 = np.argmax(XI < 0.7)
    Sxx_large = Sx1[val1:val2]
    Sxx_mean = Sxx_large.mean(axis = 1)
    max_pos = np.argmax(Sxx_mean)
    
    if compression:
        highest_freq1 = XI[val1:val2][max_pos]
        sig2, details = bandpass_filter(signal, 5, fs, 0.7, 3.5,apply_hanning=True, need_details=True)
        highest_freq2 = details['max_freq']
        highest_freq = np.min([highest_freq1, highest_freq2])
    else:
        highest_freq = XI[val1:val2][max_pos]

    mean_ibi = 1/highest_freq
    lower_freq = 1/(mean_ibi*(1.1 + band))
    upper_freq = 1/(mean_ibi*(1 - band))
    
    if lower_freq < 0.7:
        lower_freq = 0.7
    
    min_val = np.argmax(XI < upper_freq) 
    max_val = np.argmax(XI < lower_freq)
    Xi = XI[min_val:max_val]
    Sxx = Sx1[min_val :max_val, ]
    Sxy = Sxx/np.sum(Sxx, axis=0, keepdims=True)

    
    return Sxy, Xi



