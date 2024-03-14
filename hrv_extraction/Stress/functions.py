# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 15:18:10 2021

@author: Ismoil
"""
import cv2
import numpy as np
import mediapipe as mp
import scipy
from scipy.signal import welch
import math


def rgb_mean(skinFace):
    ''' This function calculates mean of the each channel for the 
    given image
    '''
    # Count how many colored pixel
    non_black_pixels_mask = np.any(skinFace != [0, 0, 0], axis=-1)
    num_pixel = np.count_nonzero(non_black_pixels_mask)

    # Count the sum for all pixels
    RGB_sum = np.sum(skinFace, axis=(0, 1))

    return RGB_sum/num_pixel


def rgb_std(skinFace):
    '''This function calculates std of the each channel for the 
    given image
    '''
    Face_skin = skinFace[np.any(skinFace != [0, 0, 0], axis=-1), :]
    return np.std(Face_skin, axis=0)


def pyVHR_face_detect(frame, sd, count, ROI_list=['face', 'forehead', 'left cheek', 'right cheek']):

    mp_face_mesh = mp.solutions.face_mesh

    forehead_pos = [66, 69, 299, 296]
    left_cheek_pos = [116, 187, 118, 216]
    right_cheek_pos = [347, 345, 436, 411]

    height, width, channels = frame.shape
    with mp_face_mesh.FaceMesh(static_image_mode=True,
                               max_num_faces=1, min_detection_confidence=0.5) as face_mesh:

        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            return 0

        x_list = []
        y_list = []
        for face in results.multi_face_landmarks:
            for landmark in face.landmark:
                x = landmark.x
                y = landmark.y

                relative_x = int(x * width)
                relative_y = int(y * height)

                x_list.append(relative_x)
                y_list.append(relative_y)

        max_x = max(x_list)
        min_x = min(x_list)

        max_y = max(y_list)
        min_y = min(y_list)
        if min_y < 0:
            min_y = 0

        face = frame[min_y:max_y, min_x:max_x, :]

        bc1 = frame[int(np.median(y_list)) - 25: int(np.median(y_list)) +
                    25, min_x - 150: min_x - 100, :]  # background 1
        bc2 = frame[int(np.median(y_list)) - 25: int(np.median(y_list)
                                                     ) + 25, max_x + 70: max_x + 120, :]  # background 2

    if count == 0:
        sd.compute_stats(face)

    skinFace, Threshold = sd.get_skin(
        face, filt_kern_size=0, verbose=False, plot=False)

    result_list = []

    for ROI in ROI_list:
        if ROI == 'face':

            result_list.append(skinFace)
        else:
            if ROI == 'forehead':
                ROI_pos = forehead_pos
            elif ROI == 'left cheek':
                ROI_pos = left_cheek_pos
            elif ROI == 'right cheek':
                ROI_pos = right_cheek_pos
            else:
                return None

            ROI_x = [x_list[i] - min_x for i in ROI_pos]
            ROI_y = [y_list[i] - min_y for i in ROI_pos]

            result = skinFace[min(ROI_y): max(
                ROI_y), min(ROI_x): max(ROI_x), :]

            result_list.append(result)

    result_list.append(bc1)
    result_list.append(bc2)

    return result_list, skinFace


def detrend(X, detLambda=10):
    # Smoothness prior approach as in the paper appendix:
    # "An advanced detrending method with application to HRV analysis"
    # by Tarvainen, Ranta-aho and Karjaalainen
    t = X.shape[0]
    l = t/detLambda  # lambda
    I = np.identity(t)
    # this works better than spdiags in python
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t-2, t)).toarray()
    detrendedX = (I-np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))).dot(X)

    return detrendedX


def preprocessing(signal, fs=30):

    # Normalize
    result = (signal - np.mean(signal))/np.std(signal)

    # Detrend
    result = detrend(result)

    return result


def PSD_comparison(RGB_list):

    RGB_list = np.array(RGB_list)

    max_list = []

    for RGB in RGB_list:

        RGB = RGB.T

        red_signal = RGB[0]

        f, Pxx_den = welch(x=red_signal, fs=30)

        max_list.append(np.max(Pxx_den))

    index = max_list.index(max(max_list))

    return RGB_list[index]


def rigrsure(x):
    x = np.array(x)  # in case that x is not an array, convert it into an array
    l = len(x)

    sx2 = [sx*sx for sx in np.absolute(x)]
    sx2.sort()
    cumsumsx2 = np.cumsum(sx2)
    risks = []
    for i in range(l):
        risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
    mini = np.argmin(risks)
    th = np.sqrt(sx2[mini])
    return th


def thresh(x, sorh, t):

    if sorh == 'hard':
        y = [e*(abs(e) >= t) for e in x]
    elif sorh == 'soft':
        y = [((e < 0)*-1.0 + (e > 0))*((abs(e)-t)*(abs(e) >= t)) for e in x]
    else:
        raise ValueError(
            'Invalid value for thresholding type, sorh = %s' % (sorh))

    return np.array(y)


def thselect(x, tptr):
    x = np.array(x)  # in case that x is not an array, convert it into an array
    l = len(x)

    if tptr == 'rigrsure':
        sx2 = [sx*sx for sx in np.absolute(x)]
        sx2.sort()
        cumsumsx2 = np.cumsum(sx2)
        risks = []
        for i in range(0, l):
            risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
        mini = np.argmin(risks)
        th = np.sqrt(sx2[mini])
    if tptr == 'heursure':
        hth = np.sqrt(2*np.log(l))

        # get the norm of x
        normsqr = np.dot(x, x)
        eta = 1.0*(normsqr-l)/l
        crit = (math.log(l, 2)**1.5)/np.sqrt(l)

        # DEBUG
#        print "crit:", crit
#        print "eta:", eta
#        print "hth:", hth
        ###

        if eta < crit:
            th = hth
        else:
            sx2 = [sx*sx for sx in abs(x)]
            sx2.sort()
            cumsumsx2 = np.cumsum(sx2)
            risks = []
            for i in range(0, l):
                risks.append((l-2*(i+1)+(cumsumsx2[i]+(l-1-i)*sx2[i]))/l)
            mini = np.argmin(risks)

            # DEBUG
#            print "risk:", risks[mini]
#            print "best:", mini
#            print "risks[222]:", risks[222]
            ###

            rth = np.sqrt(sx2[mini])
            th = min(hth, rth)
    elif tptr == 'sqtwolog':
        th = np.sqrt(2*np.log(l))
    elif tptr == 'minimaxi':
        if l < 32:
            th = 0
        else:
            th = 0.3936 + 0.1829*np.log(l, 2)
    else:
        raise ValueError(
            'Invalid value for threshold selection rule, tptr = %s' % (tptr))

    return th
