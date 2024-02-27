import json, os
import numpy as np
import json
from collections import Counter
from glob import glob

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
import pandas as pd

from downstream.calc_downstream import is_visible, is_in_cone, get_plane
from .panutils import projectPoints


def convert_coco19_to_op25(joints):
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
    op25 = np.zeros((25, 3))
    op25[1] = joints[0]  # Neck
    op25[0] = joints[1]  # Nose
    op25[8] = joints[2]  # BodyCenter
    op25[5] = joints[3]  # lShoulder
    op25[6] = joints[4]  # lElbow
    op25[7] = joints[5]  # lWrist
    op25[12] = joints[6]  # lHip
    op25[13] = joints[7]  # lKnee
    op25[14] = joints[8]  # lAnkle
    op25[2] = joints[9]  # rShoulder
    op25[3] = joints[10]  # rElbow
    op25[4] = joints[11]  # rWrist
    op25[9] = joints[12]  # rHip
    op25[10] = joints[13]  # rKnee
    op25[11] = joints[14]  # rAnkle
    op25[15] = joints[17]  # rear
    op25[16] = joints[15]  # reye
    op25[17] = joints[16]  # leye
    op25[18] = joints[18]  # lear

    return op25


def get_view_direction(joints, max_distance=10000, fov=180):
    reye, leye, rear, lear, nose, neck = joints[15], joints[16], joints[17], joints[18], joints[0], joints[1]
    # rear, reye, leye, lear, nose, neck = joints[15], joints[16], joints[17], joints[18], joints[0], joints[1]
    assert neck.sum() != 0
    if rear.sum() == 0 or lear.sum() == 0:
        view_start = (reye+leye)/2
        print('Use eyes to compute view direction')
        a, b, c, _ = get_plane(reye, leye, neck, anchor=view_start)
    else:
        assert rear.sum() != 0 and lear.sum() != 0
        view_start = (rear+lear)/2
        print('Use ears to compute view direction')
        a, b, c, _ = get_plane(rear, lear, neck, anchor=view_start)

    assert nose.sum() != 0
    assert neck.sum() != 0
    assert view_start.sum() != 0

    view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
    flag, angle, distance = is_in_cone(view_start, view_direction, nose, max_distance, fov)
    if not flag:
        print('swapping view direction:', angle, distance)
        view_direction = -view_direction
    else:
        print('not swapping view direction:', angle, distance)
    return view_start, view_direction


def plot_joints(adult_joints, infant_joints):
    # plot in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(adult_joints[:,0], adult_joints[:,1], adult_joints[:,2], c='r', marker='o')
    ax.scatter(infant_joints[:,0], infant_joints[:,1], infant_joints[:,2], c='b', marker='o')
    ax.legend(['adult', 'infant'])
    # make equal range
    max_range = np.array([adult_joints[:,0].max()-adult_joints[:,0].min(), adult_joints[:,1].max()-adult_joints[:,1].min(), adult_joints[:,2].max()-adult_joints[:,2].min()]).max() / 2.0
    mid_x = (adult_joints[:,0].max()+adult_joints[:,0].min()) * 0.5
    mid_y = (adult_joints[:,1].max()+adult_joints[:,1].min()) * 0.5
    mid_z = (adult_joints[:,2].max()+adult_joints[:,2].min()) * 0.5
    view_start_adult, view_dir_adult = get_view_direction(adult_joints)
    view_start_infant, view_dir_infant = get_view_direction(infant_joints)
    # highlight joints[0]
    ax.scatter(adult_joints[0,0], adult_joints[0,1], adult_joints[0,2], c='r', marker='o', s=20)
    ax.scatter(infant_joints[0,0], infant_joints[0,1], infant_joints[0,2], c='b', marker='o', s=20)

    # highlight joints[17, 18]
    for i in [17]:
        ax.scatter(adult_joints[i,0], adult_joints[i,1], adult_joints[i,2], c='black', marker='o', s=20)
        ax.scatter(infant_joints[i,0], infant_joints[i,1], infant_joints[i,2], c='black', marker='o', s=20)

    # highlight joints[15,16]
    for i in [1, 15]:
        ax.scatter(adult_joints[i,0], adult_joints[i,1], adult_joints[i,2], c='green', marker='o', s=20)
        ax.scatter(infant_joints[i,0], infant_joints[i,1], infant_joints[i,2], c='green', marker='o', s=20)
    

    ax.quiver(view_start_adult[0], view_start_adult[1], view_start_adult[2], view_dir_adult[0], view_dir_adult[1], view_dir_adult[2], length=0.5, normalize=True, color='r')
    ax.quiver(view_start_infant[0], view_start_infant[1], view_start_infant[2], view_dir_infant[0], view_dir_infant[1], view_dir_infant[2], length=0.5, normalize=True, color='b')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # swap y and z
    ax.view_init(azim=90, elev=-40)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # save_fig
    plt.savefig('joints_3d.png')
    

def find_adult_and_infant(gt_joints):
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

    adult_idx = np.argmax(bounding_box_size)
    infant_idx = np.argmin(bounding_box_size)
    return gt_joints[adult_idx], gt_joints[infant_idx]

# ['no touching', 'a little bit of touching', 'a little bit of touching', 'a little bit of touching', 'no touching']
seq_names = ['170915_toddler5', '160906_ian1', '160906_ian2', '160906_ian3', '160906_ian5']
data_path = '/pasteur/data/cmu_panoptic/panoptic-toolbox/scripts'
# for seq_name in seq_names:
seq_name = seq_names[0]
annotations = glob(os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19', 'body3DScene_*.json'))

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

gt_image_idx = []
gt_visibility = []
gt_touch = []
for annotation in tqdm(sorted(annotations), total=len(annotations), desc='Parsing GT labels'):
    frame_idx = int(annotation.split('/')[-1].split('.')[0].split('_')[-1])
    with open(annotation) as dfile:
        bframe = json.load(dfile)

    gt_joints = []
    for body in bframe['bodies']:
        skel = np.array(body['joints19']).reshape((-1,4))[:, :3]
        gt_joints.append(skel)
    
    if len(gt_joints) < 2:
        # print('skipping frame {} bc not enough gt joints'.format(frame_idx))
        visibility = touch = np.nan
    else:
        gt_joints = np.stack([convert_coco19_to_op25(j) for j in gt_joints])
        adult_joints_3d, infant_joints_3d = find_adult_and_infant(gt_joints)
        adult_joints_3d /= 100
        infant_joints_3d /= 100

        try:
            adult_can_see_infant, adult_angle, _ = is_visible(adult_joints_3d, infant_joints_3d, fov=180, max_distance=10000)
            infant_can_see_adult, infant_angle, _ = is_visible(infant_joints_3d, adult_joints_3d, fov=180, max_distance=10000)
            # 5 categories (adult and infant can see each other, infant can see the adult, adult can see the infant, can't see each other, NA)
            if adult_can_see_infant and infant_can_see_adult:
                visibility = 0
            elif infant_can_see_adult:
                visibility = 1
            elif adult_can_see_infant:
                visibility = 2
            else:
                visibility = 3
                
        except:
            print('skipping frame {} bc of visibility error'.format(frame_idx))
            visibility = np.nan

        joint_dist_2d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:2] - infant_joints_3d[:,:2]).sum(1)) for i in range(25)])
        joint_dist_3d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:] - infant_joints_3d).sum(1)) for i in [0]])
        touch_thresh_3d = 0.25
        touch_thresh_2d_ratio = 0.03
        image_height = 1080
        touch_2d = int(joint_dist_2d.min() > touch_thresh_2d_ratio * image_height) # 0 means within threshold!
        if not touch_2d:
            touch = int(joint_dist_3d.min() > touch_thresh_3d)
        else:
            touch = touch_2d

    # if frame_idx == 274:
    #     import pdb; pdb.set_trace()
    #     plot_joints(adult_joints_3d, infant_joints_3d)
    gt_image_idx.append(frame_idx)
    gt_visibility.append(visibility)
    gt_touch.append(touch)
gt_visibility = np.array(gt_visibility)
gt_touch = np.array(gt_touch)


def keypoints_within_frame(image_idx, cam, image_height=1080, image_width=1920):
    annotation = os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19', 'body3DScene_{:08d}.json'.format(image_idx))
    if not os.path.exists(annotation):  
        print('frame {} does not exist'.format(image_idx))
        return False

    with open(annotation) as dfile:
        bframe = json.load(dfile)

    for body in bframe['bodies']:
        gt_joints = np.array(body['joints19']).reshape((-1,4))
        keypoints_2d = projectPoints(gt_joints.transpose()[0:3, :],
                        cam['K'], cam['R'], cam['t'], 
                        cam['distCoef'])
        keypoints_2d = np.array(keypoints_2d).T  # (19, 3)

        # check whether over half of the keypoints are outside of the frame
        min_kps = 0
        if np.sum(keypoints_2d[:, 0] < 0) > min_kps or np.sum(keypoints_2d[:, 0] > image_width) > min_kps or \
                np.sum(keypoints_2d[:, 1] < 0) > min_kps or np.sum(keypoints_2d[:, 1] > image_height) > min_kps:
            # print('frame {} has more than half of keypoints outside of frame'.format(image_idx))
            return False

        # import cv2
        # image_path = os.path.join(data_path, seq_name, 'hdImgs', '00_{:02d}'.format(camera_idx), '00_{:02d}_{:08d}.jpg'.format(camera_idx, image_idx))
        # image = cv2.imread(image_path)
        # keypoints_2d = keypoints_2d.T
        # for i in range(keypoints_2d.shape[1]):
        #     cv2.circle(image, (int(keypoints_2d[0, i]), int(keypoints_2d[1, i])), 5, (0, 0, 255), -1)
        # cv2.imwrite('test.png', image)
        # import pdb; pdb.set_trace()

    return True


err_df = {
    'video': [],
    'view': [],
    'visibility': [],
    'touch': [],
    'balanced_acc_vis': [],
    'balanced_acc_touch': [],
    'support': [],
    'n_correct': []
}

for camera_idx in range(31):
    cam = cameras[(0, camera_idx)]

    pred_labels_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'cmu_panoptic_smpla', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'labels.json')
    if not os.path.exists(pred_labels_path):
        pred_labels_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'smpla_cliff', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'labels.json')
        if not os.path.exists(pred_labels_path):
            continue

    pred_visibility = [np.nan for _ in range(len(gt_visibility))]
    pred_touch = [np.nan for _ in range(len(gt_touch))]

    with open(pred_labels_path) as dfile:
        pred_labels = json.load(dfile)

    for gt_idx, image_idx in enumerate(gt_image_idx):
        image_name = 'frame_{:08d}.jpg'.format(image_idx)
        if image_name not in pred_labels:
            continue

        # if not keypoints_within_frame(image_idx, cam):
        #     continue
        label = pred_labels[image_name]

        pred_visibility[gt_idx] = label['visibility']
        pred_touch[gt_idx] = label['touch']
    
    pred_visibility = np.array(pred_visibility)
    pred_touch = np.array(pred_touch)
    valid_vis_idx = np.logical_and(~np.isnan(pred_visibility), ~np.isnan(gt_visibility))
    valid_vis_idx = np.logical_and(valid_vis_idx, pred_visibility != 4)
    valid_touch_idx = np.logical_and(~np.isnan(pred_touch), ~np.isnan(gt_touch))
    valid_touch_idx = np.logical_and(valid_touch_idx, pred_touch != 2)

    acc_vis = accuracy_score(gt_visibility[valid_vis_idx], pred_visibility[valid_vis_idx])
    acc_touch = accuracy_score(gt_touch[valid_touch_idx], pred_touch[valid_touch_idx])
    balanced_acc_vis = balanced_accuracy_score(gt_visibility[valid_vis_idx], pred_visibility[valid_vis_idx])
    balanced_acc_touch = balanced_accuracy_score(gt_touch[valid_touch_idx], pred_touch[valid_touch_idx])
    err_df['video'].append(seq_name)
    err_df['view'].append(camera_idx)
    err_df['visibility'].append(acc_vis)
    err_df['touch'].append(acc_touch)
    err_df['balanced_acc_vis'].append(balanced_acc_vis)
    err_df['balanced_acc_touch'].append(balanced_acc_touch)
    err_df['support'].append(np.sum(valid_vis_idx))
    err_df['n_correct'].append(np.sum(gt_visibility[valid_vis_idx] == pred_visibility[valid_vis_idx]))
    
    print(Counter(gt_touch[valid_touch_idx]))
    # print(camera_idx, Counter(gt_visibility[valid_vis_idx]), Counter(pred_visibility[valid_vis_idx]))
    # print(classification_report(gt_visibility[valid_vis_idx], pred_visibility[valid_vis_idx]))
    # print(classification_report(gt_touch[valid_touch_idx], pred_touch[valid_touch_idx]))
    # vis_error_frame_idx = np.where(np.logical_and(valid_vis_idx, gt_visibility != pred_visibility))[0]

err_df = pd.DataFrame(err_df)

# append mean, median, max and min, quartile of columns visibility and touch to the dataframe
vis_and_touch = err_df[['visibility', 'touch', 'balanced_acc_vis', 'balanced_acc_touch']]

# drop rows with balanced_acc_vis < vis_and_touch.mean()
bad_views = err_df[err_df['balanced_acc_vis'] < vis_and_touch.mean()['balanced_acc_vis']]['view'].values
print(bad_views)
top_err_df = err_df[err_df['balanced_acc_vis'] > vis_and_touch.mean()['balanced_acc_vis']]
top_err_df.loc['mean'] = top_err_df[['visibility', 'touch', 'balanced_acc_vis', 'balanced_acc_touch']].mean()
print(top_err_df)

err_df.loc['support'] = pd.Series(err_df['support'].sum(), index=['support'])
err_df.loc['n_correct'] = pd.Series(err_df['n_correct'].sum(), index=['n_correct'])
err_df.loc['mean'] = vis_and_touch.mean()
err_df.loc['max'] = vis_and_touch.max()
err_df.loc['min'] = vis_and_touch.min()
err_df.loc['25%'] = vis_and_touch.quantile(0.25)
err_df.loc['75%'] = vis_and_touch.quantile(0.75)
err_df.to_csv(f'labels_df_{seq_name}.csv')
print(err_df)
import pdb; pdb.set_trace()

# sbatch evaluation/cmu_panoptic/cmu_panoptic.sh

# python -m evaluation.cmu_panoptic.evaluate_downstream
# python -m evaluation.cmu_panoptic.evaluate_rec_err

