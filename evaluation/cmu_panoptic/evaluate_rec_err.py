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



class ErrAccumulator(object):
    def __init__(self):
        self.errs = []
        self.errs_for_adult = []
        self.errs_for_infant = []
        self.errs_for_joint = []

    def add(self, err, body_type):
        self.errs.append(err)
        if body_type == 'adult':
            self.errs_for_adult.append(err)
        elif body_type == 'infant':
            self.errs_for_infant.append(err)
        elif body_type == 'joint':
            self.errs_for_joint.append(err)
        else:
            raise Exception("body type {} not recognized".format(body_type))

    def get_mean(self):
        return np.mean(self.errs)
    
    def get_mean_for_adult(self):
        return np.mean(self.errs_for_adult)
    
    def get_mean_for_infant(self):
        return np.mean(self.errs_for_infant)
    
    def get_mean_for_joint(self):
        return np.mean(self.errs_for_joint)


def project_joints(joints, focal_length, principal_point):
    camera_matrix = np.array([[focal_length, 0, principal_point[0]],
                                [0, focal_length, principal_point[1]],
                                [0, 0, 1]])
    joints_2d = np.dot(camera_matrix, joints.T).T
    joints_2d = joints_2d[:, :2] / joints_2d[:, 2:]
    return joints_2d


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
downsample_factor = 30
for seq_name in seq_names:
        
    annotations = glob(os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19', 'body3DScene_*.json'))
    err_df = {
        'view': [],
        'adult_mpjpe': [],
        'infant_mpjpe': [],
        'joint_mpjpe': [],
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

    for camera_idx in range(31):
        cam = cameras[(0, camera_idx)]
        pck_for_view = {
            'adult_pck': [],
            'infant_pck': []
        }

        err_acc = ErrAccumulator()
        pred_results_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'cmu_panoptic_smpla', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'results.pt')
        pred_labels_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'cmu_panoptic_smpla', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'labels.json')

        try:
            pred_results = joblib.load(pred_results_path)
        except:
            print('skipping view {} bc no results'.format(camera_idx))
            continue
        image_name_to_results = defaultdict(list)
        for pid, res in pred_results.results.items():
            image_name_to_results[res['img_name']].append(res)

        for annotation in tqdm(sorted(annotations)[::downsample_factor], total=len(annotations[::downsample_factor]), desc='evaluating'):
            frame_idx = int(annotation.split('/')[-1].split('.')[0].split('_')[-1])
            preds = image_name_to_results['frame_{0:08d}.jpg'.format(frame_idx)]
            image_path = os.path.join(data_path, seq_name, 'hdImgs', '00_{:02d}'.format(camera_idx), '00_{:02d}_{:08d}.jpg'.format(camera_idx, frame_idx))

            body_type = [pred['body_type'] for pred in preds]
            if len(preds) < 2 or 'adult' not in body_type or 'infant' not in body_type:
                # print('skipping frame {} bc not both adult and infant'.format(frame_idx))
                continue

            with open(annotation) as dfile:
                bframe = json.load(dfile)

            gt_joints = []
            for body in bframe['bodies']:
                skel = np.array(body['joints19']).reshape((-1,4))
                gt_joints.append(skel)
            
            if len(gt_joints) < 2:
                # print('skipping frame {} bc not enough gt joints'.format(frame_idx))
                continue

            # Make the first person in gt_joints or pred_joints the adult, and the second the infant
            gt_joints = np.stack([find_matching_gt(gt_joints, pred_body_type) for pred_body_type in ["adult", "infant"]]) # (2, 19, 4)

            pred_joints = np.concatenate([pred['joints'] for pred in preds], axis=0)  # (N, 25, 3)      
            pred_transl = np.stack([pred['transl'][0] for pred in preds])      
            pred_joints = np.stack([convert_op25_to_coco19(j) for j in pred_joints])
            pred_joints = np.stack([pred_joints[np.where(np.array(body_type)==pred_body_type)[0][0]] for pred_body_type in ["adult", "infant"]]) # (2, 19, 3)

            # project gt_joints and pred_joints to image space
            # scaled_transl = pred_transl.copy()
            # scaled_transl[:, 0] *= 0.5  # x axis.
            # scaled_transl[:, 1] *= 0.3  # y axis.
            # scaled_transl[:, 2] *= 0.2
            # pred_joints[1] = pred_joints[1] - pred_transl[1] + scaled_transl[1]
            for i, body_type in enumerate(["adult", "infant"]):
                gt_keypoints_2d = projectPoints(gt_joints[i].transpose()[0:3, :],
                        cam['K'], cam['R'], cam['t'], 
                        cam['distCoef'])
                image = cv2.imread(image_path)
                # for i in range(child_keypoints_2d.shape[1]):
                #     cv2.circle(image, (int(child_keypoints_2d[0, i]), int(child_keypoints_2d[1, i])), 5, (0, 0, 255), -1)
                # overlay pred_joints
                if image is None:
                    print('skipping frame {} bc no image'.format(frame_idx))
                    pck_for_view[f'{body_type}_pck'].append(np.nan)
                    continue
                image_h, image_w, _ = image.shape
                fov = 60
                cam_focal_length = image_w / (2 * np.tan(fov * np.pi / 360))
                pred_joints_2d = project_joints(pred_joints[i], cam_focal_length, (image_w/2, image_h/2))
                # for i in range(pred_joints_2d.shape[0]):
                #     cv2.circle(image, (int(pred_joints_2d[i, 0]), int(pred_joints_2d[i, 1])), 5, (0, 255, 0), -1)
                # cv2.imwrite('test.png', image)

                pred_center = pred_joints_2d[[9, 12]].mean(0, keepdims=True).mean(0, keepdims=True)
                gt_center = gt_keypoints_2d.T[[9, 12], :2].mean(0, keepdims=True).mean(0, keepdims=True)
                pred_joints_2d = pred_joints_2d - pred_center + gt_center
                    
                diff_2d = np.linalg.norm(gt_keypoints_2d.T[:, :2] - pred_joints_2d, axis=1)
                torso_2d = np.linalg.norm(gt_keypoints_2d.T[8] - gt_keypoints_2d.T[1])
                diff_2d = diff_2d / torso_2d
                # shift the infant because of the hardcoded scaling in smpl_utils.py
                
                pck = (diff_2d < 0.3).sum() / diff_2d.shape[0]
                pck_for_view[f'{body_type}_pck'].append(pck)

            pred_center = pred_joints[:, 2:3, :3].mean(0, keepdims=True)
            gt_center = gt_joints[:, 2:3, :3].mean(0, keepdims=True)

            pred_joints -= pred_center
            gt_joints[:, :, :3] -= gt_center

            for pred_person_idx, pred_body_type in enumerate(["adult", "infant"]):
                pred_joints_ = pred_joints[pred_person_idx]
                gt_joints_ = gt_joints[pred_person_idx]

                valid_joints = np.where(gt_joints_[:15, 3] > 0.1)[0]
                # gt_joints_ = gt_joints_[valid_joints]  # only use the valid joints to compute error
                # pred_joints_ = pred_joints_[valid_joints]

                re, re_per_joint, gt_joints_hat = reconstruction_error(gt_joints_[np.newaxis, :15, :3], pred_joints_[np.newaxis, :15])
                valid_flags = (gt_joints_[:15, 3] > 0.1)
                if valid_flags.sum() > 0:
                    re = (re_per_joint * valid_flags).sum() / valid_flags.sum()
                    err_acc.add(re, pred_body_type)

            # calculate joint mpjpe
            gt_joints = gt_joints[:, :15].reshape((-1, 4))
            pred_joints = pred_joints[:, :15].reshape((-1, 3))
            re, re_per_joint, gt_joints_hat = reconstruction_error(gt_joints[np.newaxis, :, :3] / 100, pred_joints[np.newaxis, :, :3])
            valid_flags = (gt_joints[:, 3] > 0.1)
            if valid_flags.sum() > 0:
                re = (re_per_joint * valid_flags).sum() / valid_flags.sum()
                err_acc.add(re, 'joint')

        print('Overall for view {}:'.format(camera_idx))
        print(err_acc.get_mean() * 1000)
        print('adult_mpjpe:'
            '{}'.format(err_acc.get_mean_for_adult() * 1000))
        print('infant_mpjpe:'
                '{}'.format(err_acc.get_mean_for_infant() * 1000))
        print('joint_mpjpe:'
            '{}'.format(err_acc.get_mean_for_joint() * 1000))
        print('adult_pck:'
            '{}'.format(np.mean(pck_for_view['adult_pck']) * 100))
        print('infant_pck:'
            '{}'.format(np.mean(pck_for_view['infant_pck']) * 100))
        
        err_df['view'].append(camera_idx)
        err_df['adult_mpjpe'].append(err_acc.get_mean_for_adult() * 1000)
        err_df['infant_mpjpe'].append(err_acc.get_mean_for_infant() * 1000)
        err_df['joint_mpjpe'].append(err_acc.get_mean_for_joint() * 1000)
        err_df['adult_pck'].append(np.mean(pck_for_view['adult_pck']))
        err_df['infant_pck'].append(np.mean(pck_for_view['infant_pck']))

    err_df = pd.DataFrame(err_df)
    metrics = err_df.iloc[:, 1:]
    err_df.loc['90%'] = metrics.quantile(0.9)
    print(seq_name)
    print(err_df)
    err_df.to_csv(f'err_df_{seq_name}.csv', index=False)

    # sbatch evaluation/cmu_panoptic/cmu_panoptic.sh
    # python -m evaluation.cmu_panoptic.evaluate_rec_err

