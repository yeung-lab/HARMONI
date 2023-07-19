import os
import collections

import cv2
import numpy as np
from skspatial.objects import Point, Vector

# get absolute path to this file
dir_path = os.path.dirname(os.path.realpath(__file__))
sitting_pose = np.load(os.path.join(dir_path, 'anchor_poses/anchor_pose_sitting.npy'))
standing_pose = np.load(os.path.join(dir_path, 'anchor_poses/anchor_pose_standing.npy'))
supine_pose = np.load(os.path.join(dir_path, 'anchor_poses/anchor_pose_supine.npy'))
crawling_pose = np.load(os.path.join(dir_path, 'anchor_poses/anchor_pose_crawling.npy'))
# global orientation (go): rotation angle (in degrees) around x and z axis.
sitting_go = np.array([0, 180])
standing_go = np.array([0, 180])
supine_go = np.array([90, 0])
crawling_go = np.array([-90, 0])
all_gos = np.stack([sitting_go, standing_go, supine_go, crawling_go])
sitting_standing_poses = np.stack([sitting_pose, standing_pose])
focal_factor = 5
touch_thresh_3d = 0.25
touch_thresh_2d_ratio = 0.03


pose_str_map = {
    0: "Seated", 
    1: "Upright",
    2: "Supine",
    3: "Prone",
    4: "Null"
}
touch_str_map = {
    0: "Touching",
    1: "Not touching",
    2: "Null"
}
visibility_str_map = {
    0: "In each other's visible region", #"Look at each other",
    1: "Caregiver in child's visible region", # "Infant can see adult",
    2: "Child in caregiver's visible region", #"Adult can see infant",
    3: "Not in each other's visible region", # "Cannot see each other",
    4: "Null"
}

def get_valid_idxs(results, pids, filter_by_2dkp=False, min_2dkp=4):
    infant_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smpl', 'infant']]
    adult_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smplx', 'adult']]
    # filter out nan preds
    valid_infant_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in infant_pids])
    valid_adult_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in adult_pids])

    if filter_by_2dkp:
        # filter out humans where the 2D keypoints are too few
        if len(infant_pids) > 0:
            valid_infant_idxs = valid_infant_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in infant_pids])

        if len(adult_pids) > 0:
            valid_adult_idxs = valid_adult_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in adult_pids])

    valid_infant_idxs = valid_infant_idxs.astype(bool).flatten()
    valid_adult_idxs = valid_adult_idxs.astype(bool).flatten()
    
    assert len(infant_pids) == len(valid_infant_idxs)
    assert len(adult_pids) == len(valid_adult_idxs)
    return infant_pids, adult_pids, valid_infant_idxs, valid_adult_idxs


def joint_diff(joints1, joints2):
    """ Mean per-joint difference (in meters) between the two. 
    joints1: (J, 3), joints2: (J, 3)
    """
    mean_diff = np.sqrt(((joints1-joints2)**2).sum(1)).mean()
    return mean_diff

def unbatch(ts, target_dim):
    if len(ts.shape) != target_dim: 
        assert ts.shape[0] == 1
        ts = ts[0]
    assert len(ts.shape) == target_dim
    return ts


def get_plane(p1, p2, p3, anchor):
    """ 
    Returns the plane that passes through anchor, and is perpendicular to the 
    plane defined by the 3 input points.
    """
    
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
#     eye_mid = (reye+leye)/2
    x0, y0, z0 = anchor
    d = (- a * x0 - b * y0 - c * z0)
    return a, b, c, d


def is_in_cone(start_point, direction, other_point, max_distance, fov):
    """ whether the other point is in the visible cone that's defined by start_point and direction."""
    other_point = Point(other_point)
    start_point = Point(start_point)
    angle = Vector(direction).angle_between(other_point-start_point)
    distance = other_point.distance_point(start_point) 
    if np.degrees(angle) < fov/2. and distance < max_distance:
        return True, np.degrees(angle), distance
    return False, np.degrees(angle), distance


def is_visible(joints, other_joints, max_distance=1.8, fov=120):
    reye, leye, rear, lear, nose, neck = joints[15], joints[16], joints[17], joints[18], joints[0], joints[1]
    eye_mid = (reye+leye)/2
    a, b, c, _ = get_plane(rear, lear, neck, anchor=eye_mid)
    view_direction = np.array([a, b, c]) / np.linalg.norm(np.array([a, b, c]))
    if not is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]:
        view_direction = -view_direction
    assert is_in_cone(eye_mid, view_direction, nose, max_distance, fov)[0]
    for i in [15, 16, 17, 18]: # only check the other person's head joints
        flag, angle, distance = is_in_cone(eye_mid, view_direction, other_joints[i], max_distance, fov)
        if flag:
            return True, angle, distance
    return False, angle, distance  # angle of the nose


def get_valid_idxs(results, pids, filter_by_2dkp=False, min_2dkp=4):
    infant_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smil', 'infant']]
    adult_pids = [i for i in pids if i in results.results.keys() and results.results[i]['model_type'] in ['smpl', 'adult']]
    # filter out nan preds
    valid_infant_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in infant_pids])
    valid_adult_idxs = np.array([np.isnan(results.results[i]['joints']).sum() == 0 for i in adult_pids])

    if filter_by_2dkp:
        # filter out humans where the 2D keypoints are too few
        if len(infant_pids) > 0:
            valid_infant_idxs = valid_infant_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in infant_pids])

        if len(adult_pids) > 0:
            valid_adult_idxs = valid_adult_idxs & np.array([results.results[i]['keypoints'][0,:25,2].sum() >= min_2dkp for i in adult_pids])

    valid_infant_idxs = valid_infant_idxs.astype(bool).flatten()
    valid_adult_idxs = valid_adult_idxs.astype(bool).flatten()
    
    assert len(infant_pids) == len(valid_infant_idxs)
    assert len(adult_pids) == len(valid_adult_idxs)
    return infant_pids, adult_pids, valid_infant_idxs, valid_adult_idxs


def get_downstream_labels(dataset, results):
    img_to_pid = collections.defaultdict(list)
    img_list = []
    for pid, (img_name, _) in dataset.person_to_img.items():
        img_to_pid[img_name].append(pid)
        img_list.append(os.path.join(dataset.img_folder, img_name))
    img_list = np.sort(np.unique(img_list))

    n_frames = len(img_list)

    location_infant, location_adult = np.array([[np.nan]*3] * n_frames), np.array([[np.nan]*3] * n_frames).astype(np.float32)
    joints_infant, joints_adult = np.array([[[np.nan, np.nan, np.nan]]*25] * n_frames).astype(np.float32), np.array([[[np.nan, np.nan, np.nan]]*25] * n_frames).astype(np.float32)
    distance = np.array([np.nan] * n_frames).astype(np.float32)

    labels = {}
    image_height = 1080
    
    for img_idx, img_path in enumerate(img_list):
        img_name = os.path.split(img_path)[1]
        pids = img_to_pid[img_name]
        infant_pids, adult_pids, valid_infant_idxs, valid_adult_idxs = get_valid_idxs(results, pids, filter_by_2dkp=True)
        exist_dyad = sum(valid_infant_idxs) != 0 and sum(valid_adult_idxs) != 0
        calc_pose_id = sum(valid_infant_idxs) != 0

        if sum(valid_infant_idxs) != 0:
            infant_pid = np.array(infant_pids)[valid_infant_idxs][0]  # choose the first prediction to be the infant
            infant_joints = results.results[infant_pid]['joints']
            if len(infant_joints.shape) == 3: infant_joints = infant_joints[0]

            location_infant[img_idx] = infant_joints[0,:]
            joints_infant[img_idx] = infant_joints[:25]

        if sum(valid_adult_idxs) != 0: 
            if sum(valid_infant_idxs) != 0:
                # if both infant and adult are detected, choose the adult that is closer to the infant
                adult_pids_ = np.array(adult_pids)[valid_adult_idxs]
                dist_to_infant = [np.linalg.norm(results.results[adult_pid]['joints'][0] - infant_joints[0]) for adult_pid in adult_pids_]
                adult_pid = adult_pids_[np.argmin(dist_to_infant)]
            else:
                adult_pid = np.array(adult_pids)[valid_adult_idxs][0]
            adult_joints = results.results[adult_pid]['joints']
            if len(adult_joints.shape) == 3: adult_joints = adult_joints[0]

            location_adult[img_idx] = adult_joints[0,:]
            joints_adult[img_idx] = adult_joints[:25]
        
        if sum(valid_infant_idxs) != 0 and sum(valid_adult_idxs) != 0:
            distance[img_idx] = np.linalg.norm(infant_joints[:25] - adult_joints[:25]) / focal_factor

        if exist_dyad:
            infant_pid = np.array(infant_pids)[valid_infant_idxs][0]
            adult_pids = np.array(adult_pids)[valid_adult_idxs]
           
            infant_joints_3d = unbatch(results.results[infant_pid]['joints'], 2)[:25]
            
            # choose the adult that is closest to the infant
            dist_adult= [joint_diff(infant_joints_3d, unbatch(results.results[adult_pid]['joints'], 2)[:25]) for adult_pid in adult_pids]
            adult_idx = np.array(dist_adult).argmin()
            adult_pid = adult_pids[adult_idx]
            adult_joints_3d = unbatch(results.results[adult_pid]['joints'], 2)

            adult_can_see_infant, adult_angle, _ = is_visible(adult_joints_3d, infant_joints_3d, fov=200, max_distance=100)
            infant_can_see_adult, infant_angle, _ = is_visible(infant_joints_3d, adult_joints_3d, fov=200, max_distance=100.)

            # 5 categories (adult and infant can see each other, infant can see the adult, adult can see the infant, can't see each other, NA)
            if adult_can_see_infant and infant_can_see_adult:
                visibility = 0
            elif infant_can_see_adult:
                visibility = 1
            elif adult_can_see_infant:
                visibility = 2
            else:
                visibility = 3

            joint_dist_2d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:2] - infant_joints_3d[:,:2]).sum(1)) for i in range(25)])
            joint_dist_3d = np.array([np.sqrt(np.square(adult_joints_3d[[i],:] - infant_joints_3d).sum(1)) for i in [0]])
            
            touch_2d = int(joint_dist_2d.min() > touch_thresh_2d_ratio * image_height) # 0 means within threshold!
            if not touch_2d:
                touch = int(joint_dist_3d.min() > touch_thresh_3d)
            else:
                touch = touch_2d

        else:
            visibility = 4
            touch = 2
            adult_angle = infant_angle = -1
                
        if calc_pose_id:
            infant_pid = np.array(infant_pids)[valid_infant_idxs][0]
            if 'global_orient' in results.results[infant_pid].keys():
                infant_global_orient = unbatch(results.results[infant_pid]['global_orient'], 1)
                infant_pose = unbatch(results.results[infant_pid]['body_pose'], 1)
            else:
                infant_global_orient = unbatch(results.results[infant_pid]['body_pose'], 1)[:3]
                infant_pose = unbatch(results.results[infant_pid]['body_pose'], 1)[3:]
            
            joints = unbatch(results.results[infant_pid]['joints'], 2)
            keypoints = unbatch(results.results[infant_pid]['keypoints'], 2)
            legs_are_detected = keypoints[[8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24], 2].mean().item() > 0.1

            infant_neck, infant_pelvis = joints[1], joints[8]
            # angle to y axis: how tilted is the infant torso from being sitting straight.
            angle_to_y = np.degrees(np.arccos(np.abs(infant_neck - infant_pelvis)[1]/np.linalg.norm(infant_neck - infant_pelvis)))
            if angle_to_y <= 45: # sitting or standing
                # look at the thigh angle
                leg_joints = np.array([0, 1])
                rot_indices = np.concatenate([leg_joints*3, leg_joints*3+1, leg_joints*3+2])
                pose_id = np.argmin(np.linalg.norm(infant_pose[rot_indices] - sitting_standing_poses[:, rot_indices], axis=1))
            else:  # supine or crawling
                ear_mid = (joints[17] + joints[18])/2
                view_vec = Vector(joints[0]-ear_mid)
                if np.degrees(Vector(view_vec).angle_between([0,-1,0])) < 90:  
                    # y-axis is flipped, so this means it's looking upwards -> supine!
                    pose_id = 2
                else:
                    pose_id = 3

        else:
            # did not detect any infant in this frame.
            pose_id = 4
            legs_are_detected = False
            infant_to_floor = -1
            joint_dist_3d = -1

        labels[img_name] = {
            'pose': pose_id,
            'visibility': visibility,
            'touch': touch,
            'adult_angle': adult_angle,
            'infant_angle': infant_angle,
            'distance': distance[img_idx]
        }

    return labels


def add_label_per_frame(video_file, img_out_folder, img_name, pose_id, visibility, touch):
    # Add downstream labels to each frames in a video. Used for debugging.
    image_save = cv2.imread(os.path.join(img_out_folder, img_name))
    x_left = 1300
    image_save = cv2.putText(
        image_save, img_name, (20,20), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0))

    image_save = cv2.putText(
        image_save, 'Infant Pose: '+pose_str_map[pose_id], (x_left,20), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0))
    image_save = cv2.putText(
        image_save, 'Visibility: '+visibility_str_map[visibility], (x_left,50), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0))
    image_save = cv2.putText(
        image_save, 'Touch: '+touch_str_map[touch], (x_left,80), 
        cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,0))

    video_file.write(image_save)
    
