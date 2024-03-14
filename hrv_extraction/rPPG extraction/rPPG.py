"""
Created on Fri Sep 10 2021

This file contains traditional rPPG algorithms

To use this script please follow instructions below:
1. Choose an algorithm you want to use to get rPPG. (e.g. POS)
2. Call it in this manner:
    rPPG = POS(meanRGB, fps = 30, step = 5)
    
    
@author: Ismoil
"""


import numpy as np
from ICA_files import fastIca, whiten, center


def CHROM(meanRGB):
    '''
    This function calculates rPPG by using CHROM algorithm based on this paper
    {Robust pulse-rate from chrominance-based rPPG; Gerard de Haan and Vincent Jeanne}.
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels

    Output: rPPG array (N,)
    '''

    # Step 1: Compute X & Y
    Xcomp = 3*meanRGB[:, 0] - 2*meanRGB[:, 1]
    Ycomp = (1.5*meanRGB[:, 0])+meanRGB[:, 1]-(1.5*meanRGB[:, 2])

    # Step 2: Calsulate standard deviations of X & Y, as well as alpha
    sX = np.std(Xcomp)
    sY = np.std(Ycomp)

    alpha = sX/sY

    # Step 3: Compute rPPG signal
    rPPG = Xcomp-alpha*Ycomp

    return rPPG


def LGI(meanRGB):
    '''
    This function calculates rPPG by using LGI algorithm based on this paper {Local Group Invariance for Heart Rate Estimation
    from Face Videos in the Wild; Christian S. Pilz et al.}.
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels

    Output: rPPG array (N,)
    '''

    #centered = mean_RGB - np.mean(mean_RGB, axis=0)
    centered = meanRGB
    U, E, V = np.linalg.svd(centered.T)

    S = U[:, 0].T
    S = S.reshape(1, -1)

    P = np.identity(3) - (S.T @ S)

    F = (P @ centered.T).T
    rPPG = F[:, 1]
    return rPPG


def POS(meanRGB, fps, step=10):
    """
    This function calculates rPPG by using POS algorithm based on this paper {
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.} 
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels
           fps = frame rate in units of frame/s

    Output: rPPG array (N,)
    """

    eps = 10**-9
    X = meanRGB.T
    c, f = X.shape
    w = int(1.6*fps)

    Q = np.array([[0, 1, -1], [-2, 1, 1]])

    # Initialize (1)
    final_signal = np.zeros(f)
    for window_end in np.arange(w, f, step):

        window_start = window_end - w + 1

        Cn = X[:, window_start:(window_end + 1)]
        M = 1.0 / (np.mean(Cn, axis=1)+eps)
        M = np.expand_dims(M, axis=1)
        Cn = M*Cn

        S = np.dot(Q, Cn)

        S1 = S[0, :]
        S2 = S[1, :]
        alpha = np.std(S1) / (eps + np.std(S2))

        pos_signal = S1 + alpha * S2
        pos_mean = pos_signal - np.mean(pos_signal)

        final_signal[window_start:(
            window_end + 1)] = final_signal[window_start:(window_end + 1)] + pos_mean

    rPPG = final_signal
    return rPPG


def GREEN(signal):
    """
    This function calculates rPPG by using Green channel. This method is based on this paper:
    Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels

    Output: rPPG array (N,)
    """

    rPPG = signal[:, 1]
    return rPPG


def OMIT(meanRGB):
    """
    This function calculates rPPG by using OMIT algorithm based on this paper: {Álvarez Casado, C., Bordallo López, M. (2022). 
    Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces. arXiv (eprint 2202.04101).} 
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels

    Output: rPPG array (N,)
    """

    X = meanRGB
    Q, R = np.linalg.qr(X)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, X)
    rPPG = Y[1, :]

    return rPPG


def PBV(meanRGB):
    """
    This function calculates rPPG by using OMIT algorithm based on this paper: {De Haan, G., & Van Leest, A. (2014). 
    Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.} 
    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels

    Output: rPPG array (N,)
    """
    sig_mean = np.mean(meanRGB, axis=0)

    signal_norm_r = meanRGB[:, 0] / sig_mean[0]
    signal_norm_g = meanRGB[:, 1] / sig_mean[1]
    signal_norm_b = meanRGB[:, 2] / sig_mean[2]

    pbv_n = np.array([np.std(signal_norm_r), np.std(
        signal_norm_g), np.std(signal_norm_b)])
    pbv_d = np.sqrt(np.var(signal_norm_r) +
                    np.var(signal_norm_g) + np.var(signal_norm_b))
    pbv = pbv_n / pbv_d

    C = np.array([signal_norm_r, signal_norm_g, signal_norm_b])

    Q = np.matmul(C, C.T)

    W = np.linalg.solve(Q, pbv)

    A = np.matmul(C.T, np.expand_dims(W, axis=1))
    B = np.matmul(np.expand_dims(pbv, axis=1).T, np.expand_dims(W, axis=1))
    bvp = A / B
    rPPG = np.squeeze(bvp, axis=1)
    return rPPG


def ICA(meanRGB, alpha=1, thresh=1e-8, iterations=5000):
    '''
    This function calculates rPPG by using ICA algorithm based on this paper {Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). 
    Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774}    
    This function also work for 6x6 ICA. Using format is exactly same as 3x3 ICA

    Input: meanRGB array(Nx3) which is mean RGB of skinFace pixels over time; N = # of frames; 3 = RGB channels
           thresh: threshold value for error below which the loop will stop
           alpha: a hyperparameter which is a real number 
           iterations: number of iterations to be taken if thresh conditions is not fulfilled

    Output: rPPG array (N,)
    '''
    signal = meanRGB.T

    Xc, meanX = center(signal)
    Xw, whiteM = whiten(Xc)  # whiten signal
    # apply ICA and get weighting matrix
    W = fastIca(Xw,  alpha, thresh, iterations)
    unMixed = Xw.T.dot(W.T)  # find unmixed signal mean 0 version
    unMixed = (unMixed.T + meanX)  # get unmixed signal
    rPPG = unMixed[1, :]  # choose second channel as rPPG
    return rPPG
