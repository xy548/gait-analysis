# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 05:09:12 2020

@author: Xinyi

Save skeleton overlay images from video (frame by frame)
"""

import os, sys
from sys import platform
import cv2

VIDEO_ROOT_DIR = "E:\\Gait video\\"
selected_folder = ["imran"]
model_folder = "D:\\Project\\Gait\\openpose\\openpose\\models\\"

# import openpose
dir_path = os.path.dirname(os.path.realpath(__file__))
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append(dir_path + '/../../python/openpose/Release');
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.) 
        sys.path.append('../../python');
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print(
        'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


def video_to_skeleton_imgs(video_path, video, results_per_second):
    # parameters
    params = dict()
    params["model_folder"] = model_folder
    # params["logging_level"] = "4"
    kp_output_dir = os.path.join(video_path, video[:-4])
    if not os.path.exists(kp_output_dir):
        os.makedirs(kp_output_dir)

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()

    # load the video
    filepath = os.path.join(video_path, video)
    vc = cv2.VideoCapture(filepath)
    while not vc.isOpened():
        # raise FileNotFoundError('Video not found')
        vc = cv2.VideoCapture(filepath)

    c = 0
    rval = vc.isOpened()

    while rval:
        rval, frame = vc.read()
        if rval:
            # display original frame
            # cv2.imshow("original frame",frame)
            # save the original frame
            # cv2.imwrite(os.path.join(kp_output_dir, "org" + str(c) + '.jpg'), frame)
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            dest_path = os.path.join(kp_output_dir, str(c) + '.jpg')
            # display the skeleton overlay
            # cv2.imshow("original frame",datum.cvOutputData)
            cv2.imwrite(dest_path, datum.cvOutputData)
            c += 1
        else:
            break
    vc.release()


def main():
    for folder in selected_folder:
        video_path = os.path.join(VIDEO_ROOT_DIR, folder)
        videos = os.listdir(video_path)

        for video in videos:
            if ".mkv" not in video:
                # non-video file, will skip it
                continue
            video_to_skeleton_imgs(video_path, video, 30)


if __name__ == '__main__':
    main()
