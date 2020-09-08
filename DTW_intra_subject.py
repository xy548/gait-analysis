# -*- coding: utf-8 -*-
"""
Created on Mon May 25 04:02:28 2020

@author: Xinyi

DTW-SVM / KNN - intra subject - leave one out cross validation test

"""


import os
import sys
import cv2
import pandas as pd      
import numpy as np

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from utils import calculate_angles, load_skeleton, seperate_normal_abnormal, seperate_template, generate_distance_DTW


VIDEO_ROOT_DIR = "E:\\Gait video\\"
selected_folder = ["1120-01", "1120-02", "1120-03", "imran"]
# "1120-01", "1120-02","1120-03", "imran"


def main():
    for folder in selected_folder:
        filelist = os.listdir(os.path.join(VIDEO_ROOT_DIR, folder))
        normal_videos, abnormal_videos = seperate_normal_abnormal(filelist)
        template_videos, normal_videos = seperate_template(normal_videos)

        X_template = []
        for template_video in template_videos:
            X_template.append(load_skeleton(VIDEO_ROOT_DIR, folder, template_video))
        
        X_normal = []
        for normal_video in normal_videos:
            X_normal.append(load_skeleton(VIDEO_ROOT_DIR, folder, normal_video))
        X_normal = generate_distance_DTW(X_template, X_normal)
        Y_normal = np.array(["normal"] * len(normal_videos))

        X_abnormal = []
        for abnormal_video in abnormal_videos:
            X_abnormal.append(load_skeleton(VIDEO_ROOT_DIR, folder, abnormal_video))
        X_abnormal = generate_distance_DTW(X_template, X_abnormal)
        Y_abnormal = np.array(["abnormal"] * len(abnormal_videos))

        X = np.concatenate((X_normal, X_abnormal), axis=0)
        Y = np.concatenate((Y_normal, Y_abnormal), axis=0)
        SVM_predict = []
        KNN_predict = []
        
        for i in range(len(Y)):
            # i: the index of the file that is being left out for cross validation
            X_train = np.concatenate((X[:i], X[i+1:]), axis=0)
            Y_train = np.concatenate((Y[:i], Y[i+1:]), axis=0)
            X_test = np.array([X[i]])
            clf = svm.SVC(kernel='linear')
            clf.fit(X_train, Y_train)
            
            knn = KNeighborsClassifier()
            knn.fit(X_train, Y_train)
            
            SVM_predict.append(clf.predict(X_test))
            KNN_predict.append(knn.predict(X_test))

        SVM_predict = np.array(SVM_predict)
        KNN_predict = np.array(KNN_predict)

        print("----------------------------------")
        print("Current folder: ", folder)
        print('SVM precision rate:', metrics.precision_score(Y, SVM_predict,pos_label="abnormal"))
        print('SVM recall rate:', metrics.recall_score(Y, SVM_predict, pos_label="abnormal"))
        print('SVM accuracy rate :', metrics.accuracy_score(Y, SVM_predict)) 
        print('SVM F1-score:', metrics.f1_score(Y, SVM_predict, pos_label="abnormal"))
        print('SVM Confusion Matrix:', metrics.confusion_matrix(Y, SVM_predict))
        print('SVM Classification report:', metrics.classification_report(Y, SVM_predict))

        print('KNN precision rate:', metrics.precision_score(Y, KNN_predict,pos_label="abnormal"))
        print('KNN recall rate:', metrics.recall_score(Y, KNN_predict, pos_label="abnormal"))
        print('KNN accuracy rate :', metrics.accuracy_score(Y, KNN_predict)) 
        print('KNN F1-score:', metrics.f1_score(Y, KNN_predict, pos_label="abnormal"))
        print('KNN Confusion Matrix:', metrics.confusion_matrix(Y, KNN_predict))
        print('KNN Classification report:', metrics.classification_report(Y, KNN_predict))
        print("----------------------------------")

        # plt.scatter(X_normal[:,0],X_normal[:,1],color='y')      #yellowgreen
        # plt.scatter(X_abnormal[:,0],X_abnormal[:,1],color='c')  #cyan
        # plt.show()


if __name__ == '__main__':
    main()
