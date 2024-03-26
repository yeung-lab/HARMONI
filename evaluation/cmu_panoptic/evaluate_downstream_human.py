from collections import defaultdict
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


def parse_gt_annotations(path="/pasteur/data/cmu_panoptic/cmu_toddler_annotations/annotation.xlsx"):
    annotations = pd.read_excel(path)
    # filter out rows with nan
    annotations = annotations.dropna(subset=['Filename'])

    frames = annotations['Filename'].values
    visibility_all = annotations['Parent-child Visibility'].values
    touch_all = annotations['Parent-child touching'].values
    result = {}
    for frame, vis, touch in zip(frames, visibility_all, touch_all):
        if vis.lower() == "adult and infant can see each other":
            vis_label = 0
        elif vis.lower() == "infant can see the adult":
            vis_label = 1
        elif vis.lower() == "adult can see the infant":
            vis_label = 2
        elif vis.lower() == "can't see each other":
            vis_label = 3
        else:
            vis_label = 4
        
        if touch.lower() == "not touching":
            touch_label = 1
        elif touch.lower() == "touching":
            touch_label = 0
        else:
            touch_label = 2
    
        result[frame] = (vis_label, touch_label)
    return result


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

# ['no touching', 'a little bit of touching', 'a little bit of touching', 'a little bit of touching', 'no touching']
# seq_names = ['170915_toddler5', '160906_ian1', '160906_ian2', '160906_ian3', '160906_ian5']
data_path = '/pasteur/data/cmu_panoptic/panoptic-toolbox/scripts'

gt_annotations = parse_gt_annotations()
annotated_frames = gt_annotations.keys()
annotated_frames = sorted(list(annotated_frames))

seq_cam_to_frames = defaultdict(list)
for annotated_frame_name in annotated_frames:
    seq_name = '_'.join(annotated_frame_name.split('_')[:2])
    camera_idx = int(annotated_frame_name.split('_')[3])
    frame_idx = int(annotated_frame_name.split('_')[-1].split('.')[0])
    seq_cam_to_frames[(seq_name, camera_idx)].append(annotated_frame_name)


pred_visibility = []
pred_touch = []
gt_visibility = []
gt_touch = []
seq_cam_id = []  # used to filter results

for (seq_name, camera_idx), annotated_frame_names in tqdm(seq_cam_to_frames.items()):
    pred_labels_path = os.path.join('/pasteur/data/cmu_panoptic/results', 'smpla_cliff', '{}_hd_00_{:02d}'.format(seq_name, camera_idx), 'labels.json')
    if not os.path.exists(pred_labels_path):
        continue

    with open(pred_labels_path) as dfile:
        pred_labels = json.load(dfile)
    
    for annotated_frame_name in annotated_frame_names:
        frame_idx = int(annotated_frame_name.split('_')[-1].split('.')[0])
        image_name = 'frame_{0:08d}.jpg'.format(frame_idx)
        # import pdb; pdb.set_trace()
        if image_name in pred_labels:        
            label = pred_labels[image_name]
        else:
            label = {'visibility': 4, 'touch': 2}

        seq_cam_id.append((seq_name, camera_idx))
        pred_visibility.append(label['visibility'])
        pred_touch.append(label['touch'])
        gt_visibility.append(gt_annotations[annotated_frame_name][0])
        gt_touch.append(gt_annotations[annotated_frame_name][1])
        
# filter out frames with no annotations
valid_idx = np.where(np.array(pred_visibility) != 4)[0]
pred_visibility = np.array(pred_visibility)[valid_idx]
gt_visibility = np.array(gt_visibility)[valid_idx]
valid_idx = np.where(np.array(pred_touch) != 2)[0]
pred_touch = np.array(pred_touch)[valid_idx]
gt_touch = np.array(gt_touch)[valid_idx]
seq_cam_id = np.array(seq_cam_id)[valid_idx]

# keep only the best camera for each sequence
seq_names = np.unique(seq_cam_id[:, 0])
cameras = np.unique(seq_cam_id[:, 1])
acc_by_camera = {}
metric_by_seq = defaultdict(dict)
# gather accuracies by sequence and camera
for seq_name in seq_names:
    for camera in cameras:
        idx = np.where((seq_cam_id[:, 0] == seq_name) & (seq_cam_id[:, 1] == camera))[0]
        if len(idx) == 0:
            continue
        acc_vis = accuracy_score(gt_visibility[idx], pred_visibility[idx])
        acc_touch = accuracy_score(gt_touch[idx], pred_touch[idx])
        balanced_acc_vis = balanced_accuracy_score(gt_visibility[idx], pred_visibility[idx])
        balanced_acc_touch = balanced_accuracy_score(gt_touch[idx], pred_touch[idx])
        support_visibility = Counter(gt_visibility[idx])
        support_touch = Counter(gt_touch[idx])
        acc_by_camera[camera] = {
            'acc_vis': acc_vis,
            'acc_touch': acc_touch,
            'balanced_acc_vis': balanced_acc_vis,
            'balanced_acc_touch': balanced_acc_touch,
            # 'support_visibility': support_visibility,
            # 'support_touch': support_touch,
            # 'n_correct': np.sum((gt_visibility[idx] == pred_visibility[idx]) & (gt_touch[idx] == pred_touch[idx]))
        }
    # get the camera with the highest balanced accuracy
    top_k = 31//10  # 90% percentile
    
    for metric in ['acc_vis', 'acc_touch', 'balanced_acc_vis', 'balanced_acc_touch']:
        best_camera = sorted(acc_by_camera, key=lambda x: acc_by_camera[x][metric])[-top_k:]
        mean_metric = np.mean([acc_by_camera[cam][metric] for cam in best_camera])
        
        metric_by_seq[seq_name][metric] = mean_metric

print("Best results by sequence")
results = pd.DataFrame(metric_by_seq).T
# append a mean row
results.loc['mean'] = results.mean()
print(results)

#                   acc_vis  acc_touch  balanced_acc_vis  balanced_acc_touch
# 160401_ian1      0.321429   0.142857          0.357143            0.142857
# 160401_ian2      0.964286   1.000000          0.964286            1.000000
# 160401_ian3      1.000000   1.000000          1.000000            1.000000
# 170915_toddler3  1.000000   1.000000          1.000000            1.000000
# 170915_toddler4  1.000000   1.000000          1.000000            1.000000
# mean             0.857143   0.828571          0.864286            0.828571

print("Results on all frames")
# calculate accuracy and balanced accuracy
acc_vis = accuracy_score(gt_visibility, pred_visibility)
acc_touch = accuracy_score(gt_touch, pred_touch)
balanced_acc_vis = balanced_accuracy_score(gt_visibility, pred_visibility)
balanced_acc_touch = balanced_accuracy_score(gt_touch, pred_touch)
support_visibility = Counter(gt_visibility)
support_touch = Counter(gt_touch)

print('support visibility:', support_visibility)
print('support touch:', support_touch)
print('accuracy visibility:', acc_vis)
print('accuracy touch:', acc_touch)
print('balanced accuracy visibility:', balanced_acc_vis)
print('balanced accuracy touch:', balanced_acc_touch)

# sbatch evaluation/cmu_panoptic/cmu_panoptic.sh

# python -m evaluation.cmu_panoptic.evaluate_downstream_human

