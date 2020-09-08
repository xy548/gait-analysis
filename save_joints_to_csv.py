# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 03:14:20 2020

@author: Xinyi

Saving all the joints from videos into .csv files,
for convenience of data analysis
"""
import pathlib
import os
import sys
from sys import platform
import cv2
import pandas as pd
import numpy as np
from math import ceil, atan, degrees, pi, tan

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


def keypoints_distance(keypoint1, keypoint2):
    # delete confidence score
    del keypoint1[2], keypoint2[2]

    keypoint1 = np.array(keypoint1)
    keypoint2 = np.array(keypoint2)

    distance = np.sum(np.square(keypoint1 - keypoint2))
    if distance == 0:
        return 0
    else:
        distance = np.sqrt(distance)
        return distance


def video_to_matrix(video_path, video, results_per_second):
    # parameters
    params = dict()
    params["model_folder"] = model_folder
    params["logging_level"] = "4"

    kp_output_dir = os.path.join(video_path, video[:-4])
    if not os.path.exists(kp_output_dir):
        os.makedirs(kp_output_dir)

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    datum = op.Datum()

    result = []
    filepath = os.path.join(video_path, video)
    capture = cv2.VideoCapture(filepath)
    while not capture.isOpened():
        raise FileNotFoundError('Video not found')
        capture = cv2.VideoCapture(filepath)
        pass

    frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
    mid = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)
    step = int(ceil(frame_rate / results_per_second))

    # index of the output skeleton sequence
    index = 0
    # video's frame index
    video_frame_idx = 0
    # index mapping
    # key: index of the final skeleton output, value: index in the original frame
    index_frame = {}

    pre_pos = 0
    # stride length
    stride = 0
    # stride length / length of spine
    stride_ratio = 0


    while True:
        success, frame = capture.read()
        if not success:
            # print("finishing index:",index)
            # print("frames count:",video_frame_idx)
            break
        elif index % step == 0:
            video_frame_idx += 1

            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])

            # Display the frame with skeleton overlay
            # cv2.imshow("Skeleton", datum.cvOutputData)
            # cv2.waitKey(1)

            data = {'pose_keypoints_2d': datum.poseKeypoints.tolist()}

            if isinstance(data['pose_keypoints_2d'], float):
                # when skleton is not detected, data['pose_keypoints_2d'] will be a float number, instead of a list
                # print("joint data is not captured in frame {}".format(video_frame_idx))
                continue

            if len(data['pose_keypoints_2d']) > 1:
                # detected more than one person
                if pre_pos == 0:
                    # print(video_frame_idx,"situation 2")
                    continue

                compare = []
                for i in range(len(data['pose_keypoints_2d'])):
                    compare.append(abs(pre_pos - data['pose_keypoints_2d'][i][1][0]))
                data['pose_keypoints_2d'] = data['pose_keypoints_2d'][compare.index(min(compare))]

            else:
                data['pose_keypoints_2d'] = data['pose_keypoints_2d'][0]

            confidence_threshold = 0.2
            joints = {"Nose": data['pose_keypoints_2d'][0],
                      "Neck": data['pose_keypoints_2d'][1],
                      "MidHip": data['pose_keypoints_2d'][8],
                      "RAnkle": data['pose_keypoints_2d'][11],
                      "LAnkle": data['pose_keypoints_2d'][14],
                      "REar": data['pose_keypoints_2d'][17],
                      "LEar": data['pose_keypoints_2d'][18],
                      "LHeel": data['pose_keypoints_2d'][21],
                      "RHeel": data['pose_keypoints_2d'][24]
                      }

            # check the quality of the frame
            skip_flag = False
            for joint in joints:
                if joints[joint] == [0.0, 0.0, 0.0]:
                    print("missing data at joint {}, in frame: {}".format(joint, video_frame_idx))
                    skip_flag = True
                    break
                if joints[joint][2] < confidence_threshold:
                    print("low confidence at joint {}, in frame: {}".format(joint, video_frame_idx))
                    skip_flag = True

            if skip_flag:
                continue

            # update stride: distance between ankles
            cur_stride = keypoints_distance(joints["RAnkle"], joints["LAnkle"])
            if cur_stride > stride:
                stride = cur_stride
                spine = keypoints_distance(joints["Neck"], joints["MidHip"])
                stride_ratio = cur_stride / spine

            pre_pos = joints["Neck"][0]

            # Current position of the video file in milliseconds.
            time = capture.get(cv2.CAP_PROP_POS_MSEC)
            result.append({
                'time': time,
                'index': index,
                'data': data['pose_keypoints_2d']
            })
            index_frame[index] = video_frame_idx - 1
            index += 1

    if not result:
        return

    distance = []
    for i in range(len(result)):
        distance.append(abs(mid - result[i]['data'][8][0]))
    # index of the frame when the person is closest to the frame center
    center_index = distance.index(min(distance))
    # save the skeleton overlay for that frame
    capture.set(1, index_frame[center_index])
    success, frame = capture.read()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])
    dest_path = os.path.join(kp_output_dir, "center.jpg")
    cv2.imwrite(dest_path, datum.cvOutputData)
    capture.release()

    if not os.path.exists(os.path.join(kp_output_dir + "info.csv")):
        df = pd.DataFrame(columns=['index', 'stride', 'stride-spine ratio'])
        df.to_csv(kp_output_dir + "info.csv", mode='w', header=True)

    df = pd.DataFrame([[center_index, stride, stride_ratio]])
    df.to_csv(kp_output_dir + "primer.csv", mode='w', header=False, index=False)

    df = pd.DataFrame(result)
    df.to_csv(os.path.join(kp_output_dir,"jointdata.csv"), mode='w', index=False)
    # to load the data
    # joint_data = pd.read_csv(os.path.join(kp_output_dir,"jointdata.csv"))

    return result


def main():
    for folder in selected_folder:
        video_path = os.path.join(VIDEO_ROOT_DIR, folder)
        videos = os.listdir(video_path)

        for video in videos:
            if ".mkv" not in video:
                # non-video file, will skip it
                continue
            features = video_to_matrix(video_path, video, 30)


if __name__ == '__main__':
    main()
