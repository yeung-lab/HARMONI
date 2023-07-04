# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by HARMONI

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import torch
import numpy as np
from . import util
from .body import Body
from .hand import Hand
from constants import openpose_ckpts_path

class OpenposeDetector:
    def __init__(self):
        body_modelpath = os.path.join(openpose_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(openpose_ckpts_path, "hand_pose_model.pth")

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()
        with torch.no_grad():
            candidate, subset = self.body_estimation(oriImg)
            # canvas = np.zeros_like(oriImg)
            canvas = util.draw_bodypose(oriImg, candidate, subset)
            if hand:
                hands_list = util.handDetect(candidate, subset, oriImg)
                all_hand_peaks = []
                for x, y, w, is_left in hands_list:
                    peaks = self.hand_estimation(oriImg[y:y+w, x:x+w, :])
                    peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                    peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                    all_hand_peaks.append(peaks)
                canvas = util.draw_handpose(canvas, all_hand_peaks)
            return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())


def convert_to_op25(candidate, subset):
    op25 = np.zeros((25, 3))
    op25_index = [0, 1,
                  2, 3, 4,
                  5, 6, 7,
                  9, 10, 11,
                  12, 13, 14,
                  15, 16, 17, 18]

    for i in range(18):
        index = int(subset[i])
        if index == -1: continue
        if op25_index[i] == -1: continue
        op25[op25_index[i]] = candidate[index, :3]

    return op25


def run_on_images(image_paths, hand=False, vis_path=None):
    detector = OpenposeDetector()
    results = {}
    for image_path in image_paths:
        oriImg = cv2.imread(image_path)  # B,G,R order
        canvas, candidate_subset = detector(oriImg, hand)

        candidate = np.array(candidate_subset['candidate'])
        subset = np.array(candidate_subset['subset'])

        if len(subset) == 0: continue
        keypoints = []
        for i in range(len(subset)):
            openpose = convert_to_op25(candidate, subset[i])
            keypoints.append(openpose)

        if vis_path:
            cv2.imwrite(os.path.join(vis_path, os.path.basename(image_path)), canvas)

        results[image_path] = keypoints
    return results
