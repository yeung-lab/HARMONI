import json, os
import numpy as np
import json
from collections import defaultdict
from glob import glob
import pandas as pd

import cv2
import joblib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .eval_utils import reconstruction_error
from .panutils import projectPoints

def convert_op25_to_coco19(joints):
    # op25:
    #     {0,  "Nose"},
    # //     {1,  "Neck"},
    # //     {2,  "RShoulder"},
    # //     {3,  "RElbow"},
    # //     {4,  "RWrist"},
    # //     {5,  "LShoulder"},
    # //     {6,  "LElbow"},
    # //     {7,  "LWrist"},
    # //     {8,  "MidHip"},
    # //     {9,  "RHip"},
    # //     {10, "RKnee"},
    # //     {11, "RAnkle"},
    # //     {12, "LHip"},
    # //     {13, "LKnee"},
    # //     {14, "LAnkle"},
    # //     {15, "REye"},
    # //     {16, "LEye"},
    # //     {17, "REar"},
    # //     {18, "LEar"},
    # //     {19, "LBigToe"},
    # //     {20, "LSmallToe"},
    # //     {21, "LHeel"},
    # //     {22, "RBigToe"},
    # //     {23, "RSmallToe"},
    # //     {24, "RHeel"},
    # coco19: 
    # 0: Neck
    # 1: Nose
    # 2: BodyCenter (center of hips)
    # 3: lShoulder
    # 4: lElbow
    # 5: lWrist,
    # 6: lHip
    # 7: lKnee
    # 8: lAnkle
    # 9: rShoulder
    # 10: rElbow
    # 11: rWrist
    # 12: rHip
    # 13: rKnee
    # 14: rAnkle
    # 15: rear
    # 16: reye
    # 17: leye
    # 18: lear
    coco19 = np.zeros((19, 3))
    coco19[0] = joints[1]  # Neck
    coco19[1] = joints[0]  # Nose
    coco19[2] = joints[8]  # BodyCenter
    coco19[3] = joints[5]  # lShoulder
    coco19[4] = joints[6]  # lElbow
    coco19[5] = joints[7]  # lWrist
    coco19[6] = joints[12]  # lHip
    coco19[7] = joints[13]  # lKnee
    coco19[8] = joints[14]  # lAnkle
    coco19[9] = joints[2]  # rShoulder
    coco19[10] = joints[3]  # rElbow
    coco19[11] = joints[4]  # rWrist
    coco19[12] = joints[9]  # rHip
    coco19[13] = joints[10]  # rKnee
    coco19[14] = joints[11]  # rAnkle
    coco19[15] = joints[18]  # rear
    coco19[16] = joints[16]  # reye
    coco19[17] = joints[15]  # leye
    coco19[18] = joints[17]  # lear
    return coco19
    

def find_matching_gt(gt_joints, pred_body_type):
    # find bounding box of each human in gt_joints, and return the smaller one if pred_body_type is infant
    # gt_joints: (N, 19, 4)
    # pred_joints: (19, 3)
    # return: (19, 4) joints
    bounding_box_size = []
    for gt_joints_ in gt_joints:
        x = gt_joints_[:, 0]
        y = gt_joints_[:, 1]
        z = gt_joints_[:, 2]
        bounding_box_size.append((np.max(x) - np.min(x))*(np.max(y) - np.min(y))*(np.max(z) - np.min(z))/1000)

    if pred_body_type == 'adult':
        return gt_joints[np.argmax(bounding_box_size)]
    elif pred_body_type == 'infant':
        return gt_joints[np.argmin(bounding_box_size)]


def project_joints(joints, focal_length, principal_point):
    camera_matrix = np.array([[focal_length, 0, principal_point[0]],
                                [0, focal_length, principal_point[1]],
                                [0, 0, 1]])
    joints_2d = np.dot(camera_matrix, joints.T).T
    joints_2d = joints_2d[:, :2] / joints_2d[:, 2:]
    return joints_2d


def parse_human_annotations(path="/pasteur/data/cmu_panoptic/cmu_toddler_annotations/keypoints.csv"):
    # csv format
    # a.R.elbow,805,690,160401_ian1_00_01_00001721.jpg,1920,1080
    # a.R.shoulder,720,600,160401_ian1_00_01_00001721.jpg,1920,1080
    # Returns:
    # frames to [adult_keypoints, infant_keypoints]
    df = pd.read_csv(path, header=None)

    frames = np.unique(df[3])
    frames_to_keypoints = {}
    keypoint_name_to_idx = {
        'r.elbow': 10, 'l.elbow': 4,
        'r.wrist': 11, 'l.wrist': 5,
        'r.shoulder': 9, 'l.shoulder': 3,
        'r.hip': 12, 'l.hip': 6,
        'r.knee': 13, 'l.knee': 7,
        'r.ankle': 14, 'l.ankle': 8,
        'r.eye': 16, 'l.eye': 17,
    }
    for frame in frames:
        frame_df = df[df[3] == frame]
        adult_keypoints = np.zeros((19, 3))
        infant_keypoints = np.zeros((19, 3))
        for i, row in frame_df.iterrows():
            keypoint_name = row[0][2:].lower()
            if row[0].startswith('a.'):
                adult_keypoints[keypoint_name_to_idx[keypoint_name]] = [row[1], row[2], 1]
            else:
                infant_keypoints[keypoint_name_to_idx[keypoint_name]] = [row[1], row[2], 1]
        frames_to_keypoints[frame] = [adult_keypoints, infant_keypoints]
    return frames_to_keypoints


def visualize_skel(pred_joints, gt_joints):
    body_edges = np.array([[1,2],[1,4],[4,5],[5,6],[1,3],[3,7],[7,8],[8,9],[3,13],[13,14],[14,15],[1,10],[10,11],[11,12]])-1
    # visualize in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pred_joints[:,0], pred_joints[:,1], pred_joints[:,2], c='r', marker='o')
    ax.scatter(gt_joints[:,0], gt_joints[:,1], gt_joints[:,2], c='b', marker='o')
    for edge in body_edges:
        ax.plot(pred_joints[edge,0], pred_joints[edge,1], pred_joints[edge,2], c='r')
        ax.plot(gt_joints[edge,0], gt_joints[edge,1], gt_joints[edge,2], c='b')
    # add labels

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d([-100, 100])
    ax.set_ylim3d([-100, 100])
    ax.set_zlim3d([-100, 100])
    ax.legend(['pred', 'gt'])

    ax.view_init(azim=-90, elev=90)
    ax.invert_yaxis()
    plt.savefig('test.png')


seq_names = ['170915_toddler5', '160906_ian1', '160906_ian2', '160906_ian3', '160906_ian5']
data_path = '/pasteur/data/cmu_panoptic/panoptic-toolbox/scripts'
for seq_name in seq_names:
        
    annotations = glob(os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19', 'body3DScene_*.json'))
    err_df = {
        'view': [],
        'adult_pck': [],
        'infant_pck': []
    }

    # Read camera parameters
    calib_json = os.path.join(data_path, seq_name, 'calibration_{:s}.json'.format(seq_name))
    with open(calib_json, 'r') as f:
        calib = json.load(f)

    # Cameras are identified by a tuple of (panel#,node#)
    cameras = {(cam['panel'],cam['node']):cam for cam in calib['cameras']}

    # Convert data into numpy arrays for convenience
    for k,cam in cameras.items():    
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t']).reshape((3,1))

    gt_annotations = parse_human_annotations()
    annotated_frames = gt_annotations.keys()
    annotated_frames = sorted(list(annotated_frames))

    seq_cam_to_frames = defaultdict(list)
    for annotated_frame_name in annotated_frames:
        seq_name = '_'.join(annotated_frame_name.split('_')[:2])
        camera_idx = int(annotated_frame_name.split('_')[3])
        frame_idx = int(annotated_frame_name.split('_')[-1].split('.')[0])
        seq_cam_to_frames[(seq_name, camera_idx)].append(annotated_frame_name)

    for (seq_name, camera_idx), annotated_frame_names in tqdm(seq_cam_to_frames.items()):
        pred_results_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'smpla_cliff', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'results.pt')
        pred_labels_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'smpla_cliff', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'labels.json')

        try:    
            pred_results = joblib.load(pred_results_path)
        except:
            print('skipping frame {} bc no predictions'.format(frame_idx))
            continue

        image_name_to_results = defaultdict(list)
        for pid, res in pred_results.results.items():
            image_name_to_results[res['img_name']].append(res)

        for annotated_frame_name in annotated_frame_names:
            # 160401_ian1_00_01_00001721.jpg
            frame_idx = int(annotated_frame_name.split('_')[-1].split('.')[0])

            preds = image_name_to_results['frame_{0:08d}.jpg'.format(frame_idx)]
            image_path = os.path.join(data_path, seq_name, 'hdImgs', '00_{:02d}'.format(camera_idx), '00_{:02d}_{:08d}.jpg'.format(camera_idx, frame_idx))

            body_type = [pred['body_type'] for pred in preds]
            if len(preds) < 2 or 'adult' not in body_type or 'infant' not in body_type:
                # print('skipping frame {} bc not both adult and infant'.format(frame_idx))
                continue

            pred_joints = np.concatenate([pred['joints'] for pred in preds], axis=0)  # (N, 25, 3)      
            pred_transl = np.stack([pred['transl'][0] for pred in preds])      
            pred_joints = np.stack([convert_op25_to_coco19(j) for j in pred_joints])
            pred_joints = np.stack([pred_joints[np.where(np.array(body_type)==pred_body_type)[0][0]] for pred_body_type in ["adult", "infant"]]) # (2, 19, 3)

            image = cv2.imread(image_path)
            for i, body_type in enumerate(["adult", "infant"]):
                gt_keypoints_2d = gt_annotations[annotated_frame_name][i]
                image_h, image_w, _ = image.shape
                fov = 60
                cam_focal_length = image_w / (2 * np.tan(fov * np.pi / 360))
                pred_joints_2d = project_joints(pred_joints[i], cam_focal_length, (image_w/2, image_h/2))

                if gt_keypoints_2d[9, 2] > 0 and gt_keypoints_2d[12, 2] > 0:
                    pred_center = pred_joints_2d[[9, 12]].mean(0, keepdims=True).mean(0, keepdims=True)
                    gt_center = gt_keypoints_2d[[9, 12], :2].mean(0, keepdims=True).mean(0, keepdims=True)
                    pred_joints_2d = pred_joints_2d - pred_center + gt_center
                elif gt_keypoints_2d[9, 2] > 0:
                    pred_joints_2d = pred_joints_2d - pred_joints_2d[9] + gt_keypoints_2d[9, :2]
                elif gt_keypoints_2d[12, 2] > 0:
                    pred_joints_2d = pred_joints_2d - pred_joints_2d[12] + gt_keypoints_2d[12, :2]
                else:
                    gt_center = gt_keypoints_2d[gt_keypoints_2d[:, 2] > 0, :2].mean(0, keepdims=True).mean(0, keepdims=True)
                    pred_center = pred_joints_2d[gt_keypoints_2d[:, 2] > 0].mean(0, keepdims=True).mean(0, keepdims=True)
                    pred_joints_2d = pred_joints_2d - pred_center + gt_center

                valid_gt_keypoints = gt_keypoints_2d[:, 2] > 0
                if valid_gt_keypoints.sum() < 2:
                    continue
                diff_2d = np.linalg.norm(gt_keypoints_2d[valid_gt_keypoints, :2] - pred_joints_2d[valid_gt_keypoints], axis=1)
                bbox_size = np.max(gt_keypoints_2d[valid_gt_keypoints, :2], axis=0) - np.min(gt_keypoints_2d[valid_gt_keypoints, :2], axis=0)
                diff_2d = diff_2d / bbox_size.max()
                pck = (diff_2d < 0.3).sum() / diff_2d.shape[0]
                err_df[f'{body_type}_pck'].append(pck)
                
            err_df['view'].append(camera_idx)

    print('adult_pck:'
        '{}'.format(np.mean(err_df['adult_pck']) * 100))
    print('infant_pck:'
        '{}'.format(np.mean(err_df['infant_pck']) * 100))
    
    err_df = pd.DataFrame({
        'adult_pck': np.mean(err_df['adult_pck']) * 100,
        'infant_pck': np.mean(err_df['infant_pck'].mean()) * 100
    })
    err_df.to_csv(f'err_df_human_{seq_name}.csv', index=False)

    # sbatch evaluation/cmu_panoptic/cmu_panoptic.sh
    # python -m evaluation.cmu_panoptic.evaluate_rec_err_human

