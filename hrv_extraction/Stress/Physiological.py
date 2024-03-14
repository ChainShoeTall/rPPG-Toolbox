"""
Created on Wed Feb 16 12:24:01 2022
This script contains functions and classes to calculate HRV, Stress and Energy

@author: Ismoil
"""
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.signal import welch
import numpy as np



def rolling_mean(vector, size):
    running_mean = np.convolve(vector, np.ones(size)/size, mode='valid')
    return running_mean


class HRV:
    def __init__(self, ibi):

        ibi = np.array(ibi)
        # it takes RR/NN/IBI
        if ibi[-1] > 10:  # ibi should be in ms
            self.RR = ibi
            self.time = np.array([np.sum(ibi[:i])
                                 for i in range(1, len(ibi)+1)])/1000
        else:
            self.RR = ibi * 1000
            self.time = np.array([np.sum(ibi[:i])
                                 for i in range(1, len(ibi)+1)])

        self.SD = np.diff(self.RR)

    def sdnn(self):
        return np.std(self.RR)

    def rmssd(self):
        rmssd = np.sqrt(np.mean(self.SD**2))
        return rmssd

    def sdsd(self):
        return np.std(self.SD)

    def pNNxx(self, sd=50):  # pNNxx default xx = 50
        SD = abs(self.SD)
        xx = (SD > sd).sum()
        return xx/(len(self.RR) - 1)*100

    def MedSD(self):
        return np.median(abs(self.SD))

    def AMo(self):
        min_ibi, max_ibi = np.min(self.RR), np.max(self.RR)
        bins = np.arange(min_ibi, max_ibi + 51, 50)  # 50ms bin
        result = np.histogram(self.RR, bins)
        index = np.argmax(result[0])
        self.Mo = result[1][index] / 1000
        AMo = result[0][index] / len(self.RR) * 100

        return AMo

    def Mo_(self):
        self.AMo()
        return self.Mo

    def MxDMn(self):
        maxmin = max(self.RR) - min(self.RR)
        return maxmin

    def freq_domain(self, window=32, plot=False):
        RR = self.RR/1000
        tt = self.time
        dxmin = np.min(RR)
        f = interpolate.interp1d(tt, RR, 'cubic')
        xnew = np.arange(tt[0], tt[-1], dxmin)
        ynew = f(xnew)

        self.mydict = {}
        N = len(xnew)
        running_mean = rolling_mean(ynew, window)
        # high-pass signal with rolling mean
        y_signal = ynew[window//2:-(window//2 - 1)] - running_mean

        freqs, yf = welch(y_signal, 1/dxmin, nperseg=N//3)

        if plot:
            plt.plot(freqs[1:], yf[1:])
            plt.xlabel('freqs (Hz)')
            plt.ylabel('PSD ')
            plt.show()

        freq_pos = freqs[0:]
        yf_pos = yf[0:]
        # calculate VLF, LF, HF
        VLFs = freq_pos[freq_pos < 0.04]
        LF_HF = freq_pos[freq_pos >= 0.04]
        LFs = LF_HF[LF_HF < 0.15]
        HFs = LF_HF[LF_HF >= 0.15]
        HFs = HFs[HFs < 0.4]
        VLF = np.trapz(yf_pos[:len(VLFs)], VLFs)
        LF = np.trapz(yf_pos[len(VLFs):len(VLFs)+len(LFs)], LFs)
        HF = np.trapz(
            yf_pos[len(VLFs)+len(LFs):len(VLFs)+len(LFs)+len(HFs)], HFs)
        TF = VLF+LF+HF

        self.mydict['TF'] = TF*10**6
        self.mydict['VLF'] = VLF*10**6
        self.mydict['LF'] = LF*10**6
        self.mydict['HF'] = HF*10**6
        self.mydict['LF/HF'] = LF/HF

        return self.mydict

    def freq_domain_norm(self, window=32):
        self.freq_domain(window)
        dict_norm = {}
        LF = self.mydict['LF']
        HF = self.mydict['HF']
        dict_norm['LF_norm'] = LF / (LF+HF)
        dict_norm['HF_norm'] = HF / (HF+LF)
        dict_norm['LF/HF'] = self.mydict['LF/HF']
        return dict_norm


def stress_baevskiy(IBI):
    '''this function calculates Baevskiy stress index (mental stress). 
    Input: IBI- list or array of inter beat intervals.
    Output: Stress scores'''
    hrv = HRV(IBI)
    AMo = hrv.AMo()
    Mo = hrv.Mo_()
    Mxdmn = hrv.MxDMn()/1000

    SI = AMo/(2*Mo*Mxdmn)
    return np.around(SI, 1)


def stress_physical(IBI):
    '''This function calculates physical stress index
    Input: IBI- list or array of inter beat intervals.
    Output: Stress scores (range 0-100). Stress scores of 50 are optimal scores. Values less than 50 show decrease in stress
    while more than 50 show increase in stress
    '''

    hrv = HRV(IBI)
    medsd = hrv.MedSD()
    pnn50 = hrv.pNNxx()
    amo = hrv.AMo()

    mean1 = 45
    sd1 = 15
    mean2 = 35
    sd2 = 20
    mean3 = 20
    sd3 = 17

    SI = 50 + 25/3*((amo-mean1)/sd1 - (medsd-mean2)/sd2 - (pnn50-mean3)/sd3)

    if SI < 0 or SI > 100:
        print('invalid input')
        return None

    return SI


def energy(ibi):
    ''' This function calculates energy. 
    Input: IBI- list or array of inter beat intervals.
    Output: Energy scores between 0-100. Higher score means higher energy
    '''
    hrv = HRV(ibi)
    pnn50 = hrv.pNNxx()
    sdnn = hrv.SDNN()

    energy = 10.19*np.log(pnn50) + 6.73*np.log(sdnn)
    return np.round(energy)

