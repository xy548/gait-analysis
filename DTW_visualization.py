# -*- coding: utf-8 -*-
"""
Created on Mon May 25 07:31:06 2020

@author: Xinyi

Generate DTW visualization (3D)
"""


import os
import cv2
import numpy as np


import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from utils import calculate_angles, load_skeleton, seperate_normal_abnormal, seperate_template, generate_distance_DTW

VIDEO_ROOT_DIR = "E:\\Gait video\\"
selected_folder = ["1120-01", "1120-02", "1120-03", "imran"]


def main():
    X_normal = []
    X_abnormal = []
    X_template = []

    for train_folder in selected_folder:
        filelist = os.listdir(os.path.join(VIDEO_ROOT_DIR, train_folder))
        normal_videos, abnormal_videos = seperate_normal_abnormal(filelist)
        template_videos, normal_videos = seperate_template(normal_videos)

        for template_video in template_videos:
            X_template.append(load_skeleton(VIDEO_ROOT_DIR, train_folder, template_video))

        for normal_video in normal_videos:
            X_normal.append(load_skeleton(VIDEO_ROOT_DIR, train_folder, normal_video))
        
        for abnormal_video in abnormal_videos:
            X_abnormal.append(load_skeleton(VIDEO_ROOT_DIR, train_folder, abnormal_video))

    X_normal = generate_distance_DTW(X_template, X_normal)
    Y_normal = np.array(["normal"] * len(X_normal))
    
    X_abnormal = generate_distance_DTW(X_template,X_abnormal)
    Y_abnormal = np.array(["abnormal"] * len(X_abnormal))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(X_normal[:, 0], X_normal[:, 1], X_normal[:, 2], c='b', label='normal')
    ax.scatter(X_abnormal[:, 0], X_abnormal[:, 1], X_abnormal[:, 2], c='r', label='abnormal')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    plt.show()
    
    plt.scatter(X_normal[:, 0], X_normal[:, 1], c='b', label='normal')
    plt.scatter(X_abnormal[:, 0], X_abnormal[:, 1], c='r', label='abnormal')
    plt.xlabel('left lower leg angle DTW distance')
    plt.ylabel('Right lower leg angle DTW distance')
    plt.show()
    
    # plt.scatter(X_normal[:,0],X_normal[:,1],color='y')#yellowgreen
    # plt.scatter(X_abnormal[:,0],X_abnormal[:,1],color='c')#cyan
    # plt.show()
    

if __name__ == '__main__':
    main()