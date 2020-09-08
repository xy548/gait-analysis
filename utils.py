# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:21:08 2020

@author: Xinyi

BOSSVS
inter subject - Cross validation - leave one subject out experiment

"""

import json
import os
from math import atan, degrees

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from pyts.preprocessing import InterpolationImputer
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean

IMAGE_W = 1920
IMAGE_H = 1080


def calculate_angles(joint1, joint2):
    # joint1 is upper one, joint2 is lower one
    x1, y1 = joint1[0], joint1[1]
    x2, y2 = joint2[0], joint2[1]
    delta_x = abs(x1 - x2)
    delta_y = abs(y1 - y2)

    if x1 == x2:
        return 0

    angle = atan(delta_x / delta_y)
    angle = degrees(angle)

    if x1 < x2:
        return angle
    else:
        return -angle


def load_skeleton(VIDEO_ROOT_DIR, folder, video):
    """
    load the skeleton data from csv files
    :param VIDEO_ROOT_DIR: video's root directory
    :param folder: video folder
    :param video: video file name
    :return:
    """

    # number of time steps to crop
    NUM_TIMESTEPS = 130

    file_dir = os.path.join(VIDEO_ROOT_DIR, folder, video[:-4])
    joint_data = pd.read_csv(os.path.join(file_dir, "jointdata.csv"))
    joint_data["data"] = joint_data["data"].apply(json.loads)

    index_1 = len(joint_data) // 3
    index_2 = len(joint_data) // 3 * 2
    if joint_data.iloc[index_1]["data"][8][0] > joint_data.iloc[index_2]["data"][8][0]:
        # walking towards image left
        Direction = "L"
    else:
        Direction = "R"

    # flip the x coordinate, according to walk direction
    if Direction == "L":
        for i in range(len(joint_data)):
            for j in range(25):
                joint_data["data"][i][j][0] = IMAGE_W - joint_data["data"][i][j][0]

    # index in the original when it starts to have valid skeleton data
    idx_start = int(round(joint_data["time"][0] * 30 / 1000))
    # index of the last frame with valid skeleton data
    idx_end = int(round(joint_data["time"][joint_data.shape[0] - 1] * 30 / 1000))

    seq_len = idx_end - idx_start + 1

    left_seq = [None] * seq_len
    right_seq = [None] * seq_len
    back_seq = [None] * seq_len
    index_seq = [None] * seq_len

    for i in range(joint_data.shape[0]):
        # calculate the index of the original video, according to the "time" time stamp
        original_index = int(round(joint_data["time"][i] * 30 / 1000))

        # left lower leg's sequence (KNEE_LEFT, ANKLE_LEFT)
        left_seq[original_index - idx_start] = (calculate_angles(joint_data["data"][i][13], joint_data["data"][i][14]))
        # right lower leg's sequence (KNEE_RIGHT, ANKLE_RIGHT)
        right_seq[original_index - idx_start] = (calculate_angles(joint_data["data"][i][10], joint_data["data"][i][11]))
        # back angle's sequence (Neck, MidHip)
        back_seq[original_index - idx_start] = (calculate_angles(joint_data["data"][i][1], joint_data["data"][i][8]))

        # save the relative index compared to idx_start
        index_seq[original_index - idx_start] = i

    # # visualize the original left lower leg's sequence
    # plt.plot(left_seq)
    # plt.show()

    # interpolate the missing data in the sequences
    imputer = InterpolationImputer()
    impute_index = list(range(idx_start, idx_end + 1))
    left = np.array(imputer.transform([impute_index, left_seq])[1])
    right = np.array(imputer.transform([impute_index, right_seq])[1])
    back = np.array(imputer.transform([impute_index, back_seq])[1])

    # # visualize the left lower leg's sequence after interpolation
    # plt.plot(left)
    # plt.show()

    # peaks, properties = find_peaks(left, prominence=(10, None))
    # peaks_left, _ = find_peaks(left,prominence=(30, None))
    peaks_right, _ = find_peaks(right, prominence=(30, None))

    # prominences_left = peak_prominences(left, peaks_left)[0]
    # prominences_right = peak_prominences(right, peaks_right)[0]

    # peak_left_index = [index_seq[i] for i in peaks_left]
    # peak_right_index = [index_seq[i] for i in peaks_right]

    # # period of each cycle
    # T_left = [peak_left_index[i+1] - peak_left_index[i]  for i in range(len(peak_left_index)-1) ]
    # T_right = [peak_right_index[i+1] - peak_right_index[i]  for i in range(len(peak_right_index)-1) ]

    # # average period
    # T_L = sum(T_left)/len(T_left)
    # T_R = sum(T_right)/len(T_right)

    # start cropping from the first peak point of the right sequence
    start = peaks_right[0]

    # crop from the first peak point
    CROP = True

    if CROP:
        if len(joint_data) - start < NUM_TIMESTEPS:
            print("not enough data, sequence length: {}".format(len(joint_data) - start))
            return
        seqs = np.array(
            [left[start:start + NUM_TIMESTEPS], right[start:start + NUM_TIMESTEPS], back[start:start + NUM_TIMESTEPS]])

        # # blue line - left lower leg
        # plt.plot(left[start:start+NUM_TIMESTEPS])
        # plt.show()
        # # orange line - right lower leg
        # plt.plot(right[start:start+NUM_TIMESTEPS])
        # plt.plot(back[start:start+NUM_TIMESTEPS])
        # plt.ylim(-90, 90)
        # plt.show()
    else:
        if len(joint_data) < NUM_TIMESTEPS:
            print("not enough data, sequence length: {}".format(len(joint_data) - start))
            return
        seqs = np.array([left[:NUM_TIMESTEPS], right[:NUM_TIMESTEPS], back[:NUM_TIMESTEPS]])

        # # blue line - left lower leg
        # plt.plot(left[:NUM_TIMESTEPS])
        # plt.show()
        # # orange line - right lower leg
        # plt.plot(right[:NUM_TIMESTEPS])
        # plt.plot(back[:NUM_TIMESTEPS])
        # plt.ylim(-90, 90)
        # plt.show()

    return seqs


def seperate_normal_abnormal(filelist):
    normal_videos = []
    abnormal_videos = []
    for file in filelist:
        if ".mkv" not in file:
            # nonvideo file, will skip it
            continue
        if "normal" in file:
            normal_videos.append(file)
        else:
            abnormal_videos.append(file)
    return normal_videos, abnormal_videos


def seperate_template(normal_videos):
    template_videos = []
    normal = []

    num_template = 3

    for video in normal_videos:
        if "-" not in video:
            # print(video,int(video.split("normal")[1].split(".mkv")[0]))

            if int(video.split("normal")[1].split(".mkv")[0]) <= num_template:
                template_videos.append(video)
            else:
                normal.append(video)
        else:
            # in case some videos formatted with "-" inside, such as "normal-1.mkv"
            # print(video,int(video.split("normal-")[1].split(".mkv")[0]))

            if int(video.split("normal-")[1].split(".mkv")[0]) <= num_template:
                template_videos.append(video)
            else:
                normal.append(video)

    return template_videos, normal


def generate_distance_DTW(templates, ls):
    """
    generate Dynamic Time Wrapping disrance between template and the input sequence
    :param templates: reserved sequences as template
    :param ls: input list of sequences
    :return: average DTW distance
    """

    left = []
    right = []
    back = []
    # upper_left = []
    # upper_right = []

    for matrix in ls:
        left_distance = []
        right_distance = []
        back_distance = []

        for template in templates:
            l_distance, _ = fastdtw(template[0, :], matrix[0, :], dist=euclidean)
            r_distance, _ = fastdtw(template[1, :], matrix[1, :], dist=euclidean)
            b_distance, _ = fastdtw(template[2, :], matrix[2, :], dist=euclidean)

            left_distance.append(l_distance)
            right_distance.append(r_distance)
            back_distance.append(b_distance)

        left.append(sum(left_distance) / len(left_distance))
        right.append(sum(right_distance) / len(right_distance))
        back.append(sum(back_distance) / len(back_distance))

    left = np.array(left)
    right = np.array(right)
    back = np.array(back)

    return np.c_[left, right, back]
