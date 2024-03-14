"""
Last updated on Thur 29/12/2022

@author: Ismoil; PanopticAI


"""


import numpy as np
from pyampd.ampd import find_peaks
from scipy.signal import butter, sosfiltfilt, filtfilt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from kymatio.numpy import Scattering1D
from kymatio.scattering1d.filter_bank import scattering_filter_factory
from sklearn.cluster import KMeans


class Signal_Postprocessing:
    def __init__(self, rPPG, fs, interpolate = False, time_data = None, interp_freq = None):
        '''
        This class applies post-processing steps on rPPG signal to get 
        refined IBIs (needed for HRV and stress calculation) and cleaned rPPG signal.
        
        Input:
            rPPG (N,) = (array) rPPG signal of size N
            fs = (int) framerate of the video/signal
            interpolate = (bool) whether to do interpolation
            time_data (N,) = (array) time data of size N for interpolation. Time data should start from time 0sec and 
                            continue in units of seconds. It is None when interpolate = False
            interp_freq = (int) interpolation frequency. None if interpolate = False
            
        '''
        if interpolate:  
            interp_function = interp1d(time_data, rPPG, 'cubic')
            x_new = np.arange(time_data[0], time_data[-1], 1/interp_freq)
            rPPG = interp_function(x_new)
            fs = interp_freq
        
        self.fs = fs
        #self.rPPG = rPPG
        self.rPPG = self.bandpass_filter(rPPG, 50, 0.7, 7)
    
    def bandpass_filter(self, signal, order, min_freq=0.8, max_freq=4.5, apply_hanning=False, need_details=False):
        '''This function applies bandpass filter and returns filtered signal. Reccommended values are useful for Heart Rate.
        Sosfiltfilt function of SciPy locolizes the position of the peaks because it has zero phaze.

        THIS FUNCTION CANNOT BE USED IN REAL TIME


        Input: 
            signal (N,) = (array) signal of size N
            order = (int) order of butterworth signal
            min_freq = (float) min frequnecy range for HR in units of Hz
            max_freq = (float) max frequnecy range for HR in units of Hz
            apply_hanning = (bool) whether to apply hanning windowing
            need_details = (bool) whether to return details from FFT 

        Output:
            y (N,) = (array) bandpass filtered signal
            fft_details = (dictionary) FFT details such as FFT freqs, amplitudes and maximum freq in FFT range (only returned if need_details = True)

        '''
        # Step 1:  Bandpass filter
        fs = self.fs
        
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
    
    def peak_analysis(self, signal, window_size=10):
        '''
        This function calculates instantenious IBIs from given rPPG/PPG/ECG signal.

        Input:
            signal (N,) = (array) signal of size N
            window_size = (float) size of the windowing in units of number of peaks
            {Colak, A. M., Shibata, Y., & Kurokawa, F. (2016, November).
             FPGA implementation of the automatic multiscale based peak detection for real-time signal analysis on renewable energy systems.
             In 2016 IEEE International Conference on Renewable Energy Research and Applications (ICRERA) (pp. 379-384). IEEE.}

        Output:
            final_ibi (M,) = (list) IBIs in units of ms
        '''
        
        fs = self.fs
        pyamd_win = int(100/60*fs)
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
    
    def inter_cleaning(self, signal, band):
        '''
        This function is designed to clean rPPG signals. Can be used for other signals
        by adjusting parameters. Recommended values are for HR normal ranges. 
        It is part of the pipeline of HRV1.0. 

        Input:
            signal (N,) = (array) signal of size N
            band  = (float) freq band for butterworth bandpass filtering

        Output:
            finalPPG (N,) = (array) cleaned signal
        '''

    
        ppg1, mydict = self.bandpass_filter(signal, 5, 0.8, 2.5, apply_hanning=True, need_details=True)
        highest_freq = mydict['max_freq']
 

        mean_ibi = 1/highest_freq
        lower_band = 1/(mean_ibi*(1.1 + band))
        upper_band = 1/(mean_ibi*(1 - band))

        if lower_band < 0.7:
            ppg3 = self.bandpass_filter(ppg1, 4, 0.7, upper_band)
        elif upper_band > 2.5:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, 2.5)
        else:
            ppg3 = self.bandpass_filter(ppg1, 4, lower_band, upper_band)
        

        finalPPG = ppg3
        return finalPPG
    

    def HRV1_0(self, window_size=14.5, band=0.16, step=2.82):
        '''
        Input:
            window_size = (float) size of the windowing in units of seconds
            band  = (float) freq band for butterworth bandpass filtering in units of Hz
            step = (float) stepsize of the windowing in units of sec           


        Output:
            finalPPG (N,) = (array) cleaned signal
            ibi(m, ) = (array) refined IBIs
        '''
        
        
        fs = self.fs
        signal = self.rPPG
            
        N = len(signal)
        size = int(fs*window_size)
        step = int(fs*step) 
        win_ends = np.arange(size, N, step)
        L = len(win_ends)
        
        final_signal = np.zeros(N)
        coef = np.ones(L)

        for i in range(L+1):
            if i == L:
                win_end = len(signal)
            else:
                win_end = win_ends[i]

            running_window = win_end - size
            
                
            result = self.inter_cleaning(signal[running_window : win_end+1], band)
            cleaned_signal = result - np.mean(result)
            
            final_signal[running_window : win_end
                         +1] = final_signal[running_window : win_end+1] + cleaned_signal

        if L >= 2*int(size/step):
            for j in range(int(size/step)):
                coef[j] = (int(size/step) + 1) / (j + 1)
                coef[-1*j - 1] = (int(size/step) + 1) / (j + 1)
            
                if j == 0:
                    final_signal[j*step:(j+1)*step] *= coef[j]*2
                    final_signal[(-1)*step:] *= coef[j]*2
                else:
                    final_signal[j*step:(j+1)*step] *= coef[j]*1.5
                    final_signal[(-1*j-1)*step:(-1*j)*step] *= coef[j]*1.5
        else:
            pass
        
        ibi = self.peak_analysis(final_signal, window_size = 10)
        finalPPG = final_signal
        return finalPPG, np.array(ibi)
        
    
    def inter_cleaning1_2(self, signal, band, compressed = False):
        '''
        This function is designed to clean rPPG signals. It is part of the pipeline 
        for HRV1.2. Uses ScatteringTransform as basis.
        
        Input:
            signal (N,) = (array) signal of size N
            band  = (float) freq band for butterworth bandpass filtering
            compressed = (bool) whether video is compressed

        Output:
            finalPPG (N,) = (array) cleaned signal
        '''
        
        fs = self.fs
        
        
        if fs == 60 or fs == 64:
            J = 10
        elif fs == 28 or  fs == 30 or fs == 35 or fs == 32:
            J = 9
        else:
            J = 8
            
        Q = (2*J, 1)  
        T = 2**12
        
        phi_f, psi1_f, psi2_f = scattering_filter_factory(T, J, Q, T)
        XI = []
        SIGMA = []
        for psi in psi1_f:
            XI.append(psi['xi'])
            SIGMA.append(psi['sigma'])
        XI = np.array(XI)*fs
        
        scattering = Scattering1D(J, len(signal), Q, max_order = 1)
        meta = scattering.meta()
        order1 = np.where(meta['order'] == 1)
        Sx = scattering(signal)
        Sx1 = Sx[order1]
        

        val1 = np.argmax(XI < 2.5) 
        val2 = np.argmax(XI < 0.7)
        Sxx_large = Sx1[val1:val2]
        Sxx_mean = Sxx_large.mean(axis = 1)
        max_pos = np.argmax(Sxx_mean)
        
        if compressed:
            highest_freq1 = XI[val1:val2][max_pos]
            _ , details = self.bandpass_filter(signal, 5, 0.7, 3.5, apply_hanning=True, need_details=True)
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
    
    def HRV1_2(self, window_size, band, step, compressed = False):
        '''
        Input:
            window_size = (float) size of the windowing in units of seconds
            band  = (float) freq band for butterworth bandpass filtering in units of Hz
            step = (float) stepsize of the windowing in units of sec
            compressed = (bool) whether video is compressed              


        Output:
            finalPPG (N,) = (array) cleaned signal
            ibi(m, ) = (array) refined IBIs
        '''
        signal = self.rPPG
        fs = self.fs
        
        
        Sxy, Xi = self.inter_cleaning1_2(signal, band=band, compressed = compressed)
            
        N = len(signal)
        size = int(fs*window_size)
        step = int(fs*step) 
        win_ends = np.arange(size, N, step)
        L = len(win_ends)
        
        final_signal = np.zeros(N)
        coef = np.ones(L)
        
        for i in range(L+1):
            if i == L:
                win_end = len(signal)
            else:
                win_end = win_ends[i]
                
            ind = win_end//size
            rem = win_end % size
            if ind > Sxy.shape[1] - 1:
                ind = Sxy.shape[1] - 1
                energy = Sxy[:, ind] 
            else:
                energy = (size - rem)/size * Sxy[:, ind - 1] + rem/size * Sxy[:, ind]
        
        
            X = np.concatenate([Xi.reshape(-1,1), energy.reshape(-1,1)], axis = 1)
            model = KMeans(3, random_state = 0).fit(X)
            
            freqs = np.sort(model.cluster_centers_[:,0])
            lower_band = freqs[0] 
            upper_band = freqs[-1] 
        
            
            running_window = win_end - size
            
            if fs == 64 or fs == 60:
                order = 5
            else:
                order = 3
            
            
            result = self.bandpass_filter(signal[running_window : win_end+1], order, lower_band, upper_band)
            cleaned_signal = result - np.mean(result)
            
            final_signal[running_window:win_end +
                         1] = final_signal[running_window : win_end+1] + cleaned_signal
        
        
        if L >= 2*int(size/step):
            for j in range(int(size/step)):
                coef[j] = (int(size/step) + 1) / (j + 1)
                coef[-1*j - 1] = (int(size/step) + 1) / (j + 1)
            
                if j == 0:
                    final_signal[j*step:(j+1)*step] *= coef[j]*2
                    final_signal[(-1)*step:] *= coef[j]*2
                else:
                    final_signal[j*step:(j+1)*step] *= coef[j]*1.5
                    final_signal[(-1*j-1)*step:(-1*j)*step] *= coef[j]*1.5
        else:
            pass
        
        ibi = self.peak_analysis(final_signal, window_size = 10)
        finalPPG = final_signal

        return finalPPG, np.array(ibi)


    