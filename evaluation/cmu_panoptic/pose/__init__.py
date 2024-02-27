# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by HARMONI

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
import cv2
import torch
import numpy as np
from tqdm import tqdm
import json

from pose import util
from pose.body import Body
from pose.hand import Hand

class OpenposeDetector:
    def __init__(self, openpose_ckpts_path):
        body_modelpath = os.path.join(openpose_ckpts_path, "body_pose_model.pth")
        hand_modelpath = os.path.join(openpose_ckpts_path, "hand_pose_model.pth")

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    @torch.no_grad()
    def __call__(self, oriImg, hand=False, draw=False):
        oriImg = oriImg[:, :, ::-1].copy()
        candidate, subset = self.body_estimation(oriImg)

        if not draw:
            return None, dict(candidate=candidate.tolist(), subset=subset.tolist())

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


def run_on_images(detector, image_paths, hand=False, vis_path=None):
    results = {}
    for image_path in tqdm(image_paths):
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


def find_head(detector, image_paths, hand=False, vis_path=None):
    results = {}
    for image_path in tqdm(image_paths):
        image_name = os.path.basename(image_path)
        oriImg = cv2.imread(image_path)  # B,G,R order
        canvas, candidate_subset = detector(oriImg, hand)

        candidate = np.array(candidate_subset['candidate'])
        subset = np.array(candidate_subset['subset'])

        if len(subset) == 0:  # no person detected.
            results[image_name] = [0, 0, 0]
            continue

        subset = subset[0] # when there are multiple heads, use the first one.
        openpose = convert_to_op25(candidate, subset)  # (25, 3)
        head_kps = openpose[[0, 1, 15, 16, 17, 18]]
        head_kps = head_kps[head_kps[:,2] > 0.5]

        if head_kps.shape[0] > 0:
            head_center = head_kps[:, :2].mean(0).astype(int)
            results[image_name] = [int(head_center[0]), int(head_center[1]), 1]
        else: # no head detected.
            results[image_name] = [0, 0, 0]

        if vis_path:
            cv2.imwrite(os.path.join(vis_path, os.path.basename(image_path)), canvas)

    return results


def run_openpose_on_synthetic_data(image_folder):
    print('Running OpenPose on folder', image_folder)
    openpose_ckpts_path = '/sensei-fs/users/zweng/data/checkpoints'
    detector = OpenposeDetector(openpose_ckpts_path)

    humans = os.listdir(image_folder)
    print(f'{len(humans)} humans in total')
    for human_id, human in enumerate(humans):
        print(f'Processing human {human_id}', human)
        if os.path.exists(os.path.join(image_folder, human, 'uniform', 'head_centers.json')):
            print('Skip.')
            continue

        rendering_path = os.path.join(image_folder, human, 'uniform/renderings')
        if not os.path.exists(rendering_path):
            print('Renderings empty')
            continue
        image_paths = [os.path.join(rendering_path, filename) for filename in os.listdir(rendering_path) if filename.endswith('_rgb.png')]
        image_paths = sorted(image_paths)
        head_centers = find_head(detector, image_paths, hand=False, vis_path=None)

        with open(os.path.join(image_folder, human, 'uniform', 'head_centers.json'), 'w') as f:
            json.dump(head_centers, f)


def run_openpose_on_humman_data(image_folder, start_idx):
    # /sensei-fs/users/zweng/data/HumanNeRF/HuMMan_V1.0/raw/p000579_a000396/kinect_color/kinect_000/000024.png
    openpose_ckpts_path = '/sensei-fs/users/zweng/data/checkpoints'
    detector = OpenposeDetector(openpose_ckpts_path)

    sequences = os.listdir(image_folder)
    print(f'{len(sequences)} sequences in total')
    for sequence_id, sequence in enumerate(sequences[start_idx:]):
        print(f'Processing {sequence_id}, sequence {sequence}')
        if os.path.exists(os.path.join(image_folder, sequence, 'head_centers.json')):
            print('Skip.')
            continue

        head_centers_all_views = {}
        view_ids = os.listdir(os.path.join(image_folder, sequence, 'kinect_color'))  # ['kinect_000', 'kinect_001', ...]
        for view_id in view_ids:
            rendering_path = os.path.join(image_folder, sequence, 'kinect_color', view_id)
            image_paths = [os.path.join(rendering_path, filename) for filename in os.listdir(rendering_path) if filename.endswith('.png')]
            image_paths = sorted(image_paths)
            head_centers = find_head(detector, image_paths, hand=False, vis_path=None)

            head_centers = {os.path.join(view_id, k): v for k, v in head_centers.items()}
            head_centers_all_views.update(head_centers)

        with open(os.path.join(image_folder, sequence, 'head_centers.json'), 'w') as f:
            json.dump(head_centers_all_views, f) 


if __name__ == '__main__':
    # about 2.3 seconds per image.
    image_folder = sys.argv[1]
    start_idx = int(sys.argv[2])

    # Example usage: python -m pose.__init__ '/sensei-fs/users/zweng/data/training_data_0elev/THuman_render18'
    run_openpose_on_synthetic_data(image_folder)

    # Example usage: python -m pose.__init__ '/sensei-fs/users/zweng/data/HumanNeRF/HuMMan_V1.0/raw' 100
    # run_openpose_on_humman_data(image_folder, start_idx)
