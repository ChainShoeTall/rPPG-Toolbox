"""
Created on Tue May  3 10:33:05 2022

@author: Ismoil

This file reads videos and ground truth DATA from and returns NPZ file that contains
PPG_gt, SPO_gt, HR_gt, rPPG (all with time steps), an istant of face from video
"""

import pandas as pd
import numpy as np
import cv2
from datetime import datetime
import os
from functions import pyVHR_face_detect, rgb_mean
from resources.SkinDetect import SkinDetect
from multiprocessing import Pool


def video_synch(file_name):
    PATH = "D:\\dataset\\StressData\\Stress_final_setup"

    # read GROUND TRUTH data
    time_file = PATH + '/' + file_name + '/' + file_name + \
        '_times.csv'  # when each experimental part starts and ends
    frames_file = PATH + '/' + file_name + '/' + file_name + "_frames.csv"
    video_file = PATH + '/' + file_name + '/' + file_name + ".avi"
    gt_file = PATH + '/' + file_name + '/' + file_name + "_CONTEC.csv"

    times = np.array(pd.read_csv(time_file))
    for i in range(len(times)):
        times[i, 1] = datetime.strptime(times[i, 1], "%d/%m/%Y, %H:%M:%S.%f")

    frames = np.array(pd.read_csv(frames_file))
    for i in range(len(frames)):
        frames[i, 1] = datetime.strptime(frames[i, 1], "%d/%m/%Y, %H:%M:%S.%f")

    gt = np.array(pd.read_csv(gt_file))
    for i in range(len(gt)):
        gt[i, 0] = datetime.strptime(gt[i, 0], "%d/%m/%Y, %H:%M:%S.%f")

    # Find where in the video each experimental part starts and ends
    train_start = np.argmin(abs(times[0, 1] - frames[:, 1]))
    train_end = np.argmin(abs(times[1, 1] - frames[:, 1]))

    test1_start = np.argmin(abs(times[4, 1] - frames[:, 1]))
    test1_end = np.argmin(abs(times[5, 1] - frames[:, 1]))

    test2_start = np.argmin(abs(times[2, 1] - frames[:, 1]))
    test2_end = np.argmin(abs(times[3, 1] - frames[:, 1]))

    # Find where in the CONTEC file each experimental part starts and ends
    train_start_gt = np.argmin(abs(times[0, 1] - gt[:, 0]))
    train_end_gt = np.argmin(abs(times[1, 1] - gt[:, 0]))

    test1_start_gt = np.argmin(abs(times[4, 1] - gt[:, 0]))
    test1_end_gt = np.argmin(abs(times[5, 1] - gt[:, 0]))

    test2_start_gt = np.argmin(abs(times[2, 1] - gt[:, 0]))
    test2_end_gt = np.argmin(abs(times[3, 1] - gt[:, 0]))

    train_gt = gt[train_start_gt:train_end_gt, [0, 6, 11, 13]]
    test1_gt = gt[test1_start_gt:test1_end_gt, [0, 6, 11, 13]]
    test2_gt = gt[test2_start_gt:test2_end_gt, [0, 6, 11, 13]]

    cap = cv2.VideoCapture(video_file)
    sd = SkinDetect(strength=0.2)
    i = 0
    RGB_list0 = np.zeros([train_end - train_start, 3])
    RGB_list1 = np.zeros([test1_end - test1_start, 3])
    RGB_list2 = np.zeros([test2_end - test2_start, 3])

    dictionary = {}
    dictionary["train_gt"] = train_gt
    dictionary["test1_gt"] = test1_gt
    dictionary["test2_gt"] = test2_gt

    while cap.isOpened():  # start loop
        ret, frame = cap.read()

        if not ret:
            break

        if ret:

            if i == 60:  # this skin detection algorithm needs one frame to process and make statistics
                _, skinFace = pyVHR_face_detect(frame, sd, 0)
                dictionary['skinFace'] = skinFace
                dictionary['frame'] = frame

            # cv2.imshow('frame', frame)

            if i >= train_start and i < train_end:  # process NO STRESS part. Obtain rPPG
                _, skinFace = pyVHR_face_detect(frame, sd, i)
                b, g, r = rgb_mean(skinFace)
                RGB_list0[i-train_start, 0], RGB_list0[i-train_start,
                                                       1], RGB_list0[i-train_start, 2] = r, g, b

            elif i >= test1_start and i < test1_end:  # process TEST1 part. Obtain rPPG
                _, skinFace = pyVHR_face_detect(frame, sd, i)
                b, g, r = rgb_mean(skinFace)
                RGB_list1[i-test1_start, 0], RGB_list1[i-test1_start,
                                                       1], RGB_list1[i-test1_start, 2] = r, g, b

            elif i >= test2_start and i < test2_end:  # process TEST2 part. Obtain rPPG
                _, skinFace = pyVHR_face_detect(frame, sd, i)
                b, g, r = rgb_mean(skinFace)
                RGB_list2[i-test2_start, 0], RGB_list2[i-test2_start,
                                                       1], RGB_list2[i-test2_start, 2] = r, g, b

            i += 1
            if cv2.waitKeyEx(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    dictionary["meanRGB_train"] = np.concatenate((RGB_list0,
                                                  frames[train_start:train_end, 1].reshape(-1, 1)), axis=1)
    dictionary["meanRGB_test1"] = np.concatenate((RGB_list1,
                                                  frames[test1_start:test1_end, 1].reshape(-1, 1)), axis=1)
    dictionary["meanRGB_test2"] = np.concatenate((RGB_list2,
                                                  frames[test2_start:test2_end, 1].reshape(-1, 1)), axis=1)

    np.savez(PATH + '/' + file_name + '/' + file_name + '.npz', dictionary)

    return 0

###############################################################################


'''
PATH = "D:\\dataset\\StressData\\Stress_final_setup"
files = os.listdir(PATH)

if __name__ == "__main__":
    pp = Pool()
    pp.map(video_synch, files)
    pp.close()
    pp.join()
'''
