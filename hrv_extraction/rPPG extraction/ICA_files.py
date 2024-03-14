# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 14:55:06 2021

@author: Ismoil
"""
import numpy as np


def center(x):
    mean = np.mean(x, axis=1, keepdims=True)
    centered = x - mean
    return centered, mean


def covariance(x):
    mean = np.mean(x, axis=1, keepdims=True)
    n = np.shape(x)[1] - 1
    m = x - mean

    return (m.dot(m.T))/n


def whiten(x):
    # Calculate the covariance matrix
    coVarM = covariance(x)

    # Single value decoposition
    U, S, V = np.linalg.svd(coVarM)

    # Calculate diagonal matrix of eigenvalues
    d = np.diag(1.0 / np.sqrt(S))

    # Calculate whitening matrix
    whiteM = np.dot(U, np.dot(d, U.T))

    # Project onto whitening matrix
    Xw = np.dot(whiteM, x)

    return Xw, whiteM


def fastIca(signals,  alpha=1, thresh=1e-8, iterations=5000):
    m, n = signals.shape

    # Initialize random weights
    W = np.random.rand(m, m)

    for c in range(m):
        w = W[c, :].copy().reshape(m, 1)
        w = w / np.sqrt((w ** 2).sum())

        i = 0
        lim = 100
        while ((lim > thresh) & (i < iterations)):

            # Dot product of weight and signal
            ws = np.dot(w.T, signals)

            # Pass w*s into contrast function g
            wg = np.tanh(ws * alpha).T

            # Pass w*s into g prime
            wg_ = (1 - np.square(np.tanh(ws))) * alpha

            # Update weights
            wNew = (signals * wg.T).mean(axis=1) - wg_.mean() * w.squeeze()

            # Decorrelate weights
            wNew = wNew - np.dot(np.dot(wNew, W[:c].T), W[:c])
            wNew = wNew / np.sqrt((wNew ** 2).sum())

            # Calculate limit condition
            lim = np.abs(np.abs((wNew * w).sum()) - 1)

            # Update weights
            w = wNew

            # Update counter
            i += 1

        W[c, :] = w.T
    return W
