# -*- coding: utf-8 -*-
"""
Created on Sat May 23 02:59:51 2020

@author: Xinyi

feature extraction

"""

import os
import pandas as pd
import numpy as np
from math import ceil, atan, degrees
import json

from pyts.preprocessing import InterpolationImputer

from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from scipy import stats

from utils import calculate_angles, seperate_normal_abnormal, seperate_template, generate_distance_DTW

VIDEO_ROOT_DIR = "E:\\Gait video\\"
selected_folder = ["1120-01", "1120-02", "1120-03", "imran"]
# "1120-01", "1120-02","1120-03", "imran"

IMAGE_W = 1920
IMAGE_H = 1080


def calculate_angles_horizontal(joint1, joint2):
    # joint1 is left one, joint2 is right one
    x1 = joint1[0]
    y1 = joint1[1]
    x2 = joint2[0]
    y2 = joint2[1]
    delta_x = abs(x1 - x2)
    delta_y = abs(y1 - y2)

    if y1 == y2:
        return 0

    angle = atan(delta_y / delta_x)
    angle = degrees(angle)

    if y1 < y2:
        return angle
    else:
        return -angle


def load_skeleton(VIDEO_ROOT_DIR, folder, video):
    file_dir = os.path.join(VIDEO_ROOT_DIR, folder, video[:-4])
    joint_data = pd.read_csv(os.path.join(file_dir, "jointdata.csv"))
    joint_data["data"] = joint_data["data"].apply(json.loads)
    index = pd.read_csv(file_dir + "primer.csv", header=None)[0][0]

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

    # calculate speed    pixel/second    
    Neck_1 = joint_data["data"][index_1][1]
    Neck_2 = joint_data["data"][index_2][1]
    MidHip_1 = joint_data["data"][index_1][8]
    MidHip_2 = joint_data["data"][index_2][8]
    delta_x_OP = (Neck_2[0] + MidHip_2[0]) / 2 - (Neck_1[0] + MidHip_1[0]) / 2
    delta_t = abs(joint_data["time"][index_2] - joint_data["time"][index_1]) / 1000
    if delta_t == 0:
        print("error in speed calculation")
        return -1
    else:
        SpeedOPose = abs(delta_x_OP / delta_t)

    # calculate 3 angles
    Neck = joint_data["data"][index][1]
    MidHip = joint_data["data"][index][8]
    Nose = joint_data["data"][index][0]

    Body_Lean = calculate_angles(Nose, MidHip)
    Back_Lean = calculate_angles(Neck, MidHip)
    Neck_Lean = calculate_angles(Nose, Neck)

    # index in the original when it starts to have valid skeleton data
    idx_start = int(round(joint_data["time"][0] * 30 / 1000))
    # index of the last frame with valid skeleton data
    idx_end = int(round(joint_data["time"][joint_data.shape[0] - 1] * 30 / 1000))

    seq_len = idx_end - idx_start + 1

    left_seq = [None] * seq_len
    right_seq = [None] * seq_len
    index_seq = [None] * seq_len

    for i in range(joint_data.shape[0]):
        # calculate the index of the original video, according to the "time" time stamp
        original_index = int(round(joint_data["time"][i] * 30 / 1000))

        # left lower leg's sequence (KNEE_LEFT, ANKLE_LEFT)
        left_seq[original_index - idx_start] = calculate_angles(joint_data["data"][i][13], joint_data["data"][i][14])
        # right lower leg's sequence (KNEE_RIGHT, ANKLE_RIGHT)
        right_seq[original_index - idx_start] = calculate_angles(joint_data["data"][i][10], joint_data["data"][i][11])

        # save the relative index compared to idx_start
        index_seq[original_index - idx_start] = i

    imputer = InterpolationImputer()
    impute_index = list(range(idx_start, idx_end + 1))
    left = np.array(imputer.transform([impute_index, left_seq])[1])
    right = np.array(imputer.transform([impute_index, right_seq])[1])
    # plt.plot(impute_index,left_seq)
    # plt.show()

    # plt.plot(impute_index,left)
    # plt.show()

    peaks_left, bottom_left = sequence_extrema(left)
    peaks_right, bottom_right = sequence_extrema(right)

    left_stance, left_swing = stance_swing(peaks_left, bottom_left)
    right_stance, right_swing = stance_swing(peaks_right, bottom_right)

    # Asymmetry Stance phase
    AStP = calculate_asymmetry(left_stance, right_stance)
    # Asymmetry Swing phase
    ASwP = calculate_asymmetry(left_swing, right_swing)

    cadence = calculate_cadence(peaks_left, peaks_right)

    left_peak_amp = np.mean([left[i] for i in peaks_left])
    left_bottom_amp = np.mean([left[i] for i in bottom_left])
    right_peak_amp = np.mean([right[i] for i in peaks_right])
    right_bottom_amp = np.mean([right[i] for i in bottom_right])

    # Asymmetry Peak Amplitude
    APA = calculate_asymmetry(left_peak_amp, right_peak_amp)
    # Asymmetry Bottom Amplitude
    ABA = calculate_asymmetry(left_bottom_amp, right_bottom_amp)

    L_index = ceil(len(peaks_left) / 2) - 1
    R_index = ceil(len(peaks_right) / 2) - 1

    # left lower leg:  peaks_left[L_index] to peaks_left[L_index+1]
    # right lower leg:  peaks_right[R_index] to peaks_right[R_index+1]
    # step length stride length: distance between the heel contact point of one foot and that of the other foot.

    left_step_length = abs(joint_data["data"][index_seq[peaks_left[L_index]]][21][0] -
                           joint_data["data"][index_seq[peaks_left[L_index]]][24][0])
    right_step_length = abs(joint_data["data"][index_seq[peaks_left[R_index]]][21][0] -
                            joint_data["data"][index_seq[peaks_left[R_index + 1]]][24][0])

    # Asymmetry Step length
    ASl = calculate_asymmetry(left_step_length, right_step_length)

    stride_length = abs(joint_data["data"][index_seq[peaks_left[L_index]]][21][0] -
                        joint_data["data"][index_seq[peaks_right[L_index]]][21][0])

    falling_risk = abs(joint_data["data"][index_seq[peaks_left[L_index]]][0][0] - (
            joint_data["data"][index_seq[peaks_left[L_index]]][21][0] +
            joint_data["data"][index_seq[peaks_left[L_index]]][24][0]) / 2) / (
                           abs(joint_data["data"][index_seq[peaks_left[L_index]]][19][0] -
                               joint_data["data"][index_seq[peaks_left[L_index]]][24][0]) / 2)

    features = [SpeedOPose, Body_Lean, Back_Lean, Neck_Lean,
                left_stance, left_swing, right_stance, right_swing, AStP, ASwP,
                cadence, left_peak_amp, right_peak_amp,
                left_bottom_amp, right_bottom_amp, APA, ABA,
                left_step_length, right_step_length, ASl,
                stride_length, falling_risk]

    return features


def sequence_extrema(seq):
    # maxima 
    peaks, _ = find_peaks(seq, prominence=(30, None))
    # minima
    bottoms, _ = find_peaks(-seq, prominence=(30, None))

    if bottoms[0] < peaks[0]:
        bottoms = bottoms[1:]
    if bottoms[-1] > peaks[-1]:
        bottoms = bottoms[:-1]
    # print(peaks)
    # print(bottoms)
    # print("================")
    return peaks, bottoms


def stance_swing(peaks, bottoms):
    # calculate stance phase and swing phase
    if len(peaks) - len(bottoms) != 1:
        print("missing or wrong data")

    stance = []
    swing = []

    for i in range(len(bottoms)):
        stance.append(bottoms[i] - peaks[i])
        swing.append(peaks[i + 1] - bottoms[i])

    # print(stance)
    # print(swing)
    # time in seconds(unit)
    return np.mean(stance) / 30, np.mean(swing) / 30


def calculate_cadence(peaks_left, peaks_right):
    if len(peaks_left) != len(peaks_right):
        print("mismatch")
        if len(peaks_left) > len(peaks_right):
            peaks_left = peaks_left[:len(peaks_right)]
        else:
            peaks_right = peaks_right[:len(peaks_left)]

    diff = []

    for i in range(len(peaks_right)):
        diff.append(peaks_left[i] - peaks_right[i])

    # step time in seconds
    step_time = np.mean(diff) / 30

    cadence = 60 / step_time
    return cadence


def calculate_asymmetry(fA, fB):
    Af = abs(fA - fB) / max(fA, fB)
    return Af


def main():
    for folder in selected_folder:
        filelist = os.listdir(os.path.join(VIDEO_ROOT_DIR, folder))
        normal_videos, abnormal_videos = seperate_normal_abnormal(filelist)

        X_normal = []
        for normal_video in normal_videos:
            X_normal.append(load_skeleton(VIDEO_ROOT_DIR, folder, normal_video))
        X_normal = np.array(X_normal)
        Y_normal = np.array(["normal"] * len(normal_videos))

        X_abnormal = []
        for abnormal_video in abnormal_videos:
            X_abnormal.append(load_skeleton(VIDEO_ROOT_DIR, folder, abnormal_video))
        X_abnormal = np.array(X_abnormal)
        Y_abnormal = np.array(["abnormal"] * len(abnormal_videos))

        X = np.concatenate((X_normal, X_abnormal), axis=0)
        Y = np.concatenate((Y_normal, Y_abnormal), axis=0)

        data = pd.DataFrame(X, columns=["SpeedOPose", "Body_Lean", "Back_Lean", "Neck_Lean",
                                        "left_stance", "left_swing", "right_stance", "right_swing", "AStP", "ASwP",
                                        "cadence", "left_peak_amp", "right_peak_amp",
                                        "left_bottom_amp", "right_bottom_amp", "APA", "ABA",
                                        "left_step_length", "right_step_length", "ASl",
                                        "stride_length", "falling_risk"])
        label = pd.Series(Y)
        data["type"] = label
        data.to_excel(folder + '.xlsx')

        normal = data.loc[data["type"] == "normal"]
        abnormal = data.loc[data["type"] == "abnormal"]

        if not os.path.exists(os.path.join(VIDEO_ROOT_DIR, folder, "box_plot")):
            os.makedirs(os.path.join(VIDEO_ROOT_DIR, folder, "box_plot"))

        for feature in list(data.columns):
            if feature == "type":
                continue
            print("--------------------------")
            print(feature)
            print(stats.ttest_ind(normal[feature], abnormal[feature]))
            print("--------------------------")

            fig, ax = plt.subplots()
            ax.set_title(feature)
            ax.set_xticklabels(['normal gait', 'abnormal gait'])
            ax.boxplot([normal[feature], abnormal[feature]])
            plt.savefig(os.path.join(VIDEO_ROOT_DIR, folder, "box_plot", feature + ".png"))
            plt.show()
            # data.boxplot(column = feature, by='type',return_type=None)


if __name__ == '__main__':
    main()
