import cv2
import numpy as np


def get_openpose_joint_names():
    return [
        'OP Nose',        # 0
        'OP Neck',        # 1
        'OP RShoulder',   # 2
        'OP RElbow',      # 3
        'OP RWrist',      # 4
        'OP LShoulder',   # 5
        'OP LElbow',      # 6
        'OP LWrist',      # 7
        'OP MidHip',      # 8
        'OP RHip',        # 9
        'OP RKnee',       # 10
        'OP RAnkle',      # 11
        'OP LHip',        # 12
        'OP LKnee',       # 13
        'OP LAnkle',      # 14
        'OP REye',        # 15
        'OP LEye',        # 16
        'OP REar',        # 17
        'OP LEar',        # 18
        'OP LBigToe',     # 19
        'OP LSmallToe',   # 20
        'OP LHeel',       # 21
        'OP RBigToe',     # 22
        'OP RSmallToe',   # 23
        'OP RHeel',       # 24
    ]


def get_openpose_skeleton():
    return np.array(
        [
            [0 , 1],
            [1 , 2],
            [2 , 3],
            [3 , 4],
            [1 , 5],
            [5 , 6],
            [6 , 7],
            [1 , 8],
            [8 , 9],
            [9 ,10],
            [10,11],
            [8 ,12],
            [12,13],
            [13,14],
            [0 ,15],
            [0 ,16],
            [15,17],
            [16,18],
            [21,19],
            [19,20],
            [14,21],
            [11,24],
            [24,22],
            [22,23],
        ]
    )


def draw_skeleton(image, kp_2d, dataset='openpose', unnormalize=True,
                  thickness=2, res=224, j_error=None, j_uncertainty=None, print_joints=False):

    if np.max(image) < 10:
        image = image * 255
        image = np.clip(image, 0, 255)
        image = np.asarray(image, dtype=np.uint8)

    if unnormalize:
        kp_2d[:,:2] = 0.5 * res * (kp_2d[:, :2] + 1) # normalize_2d_kp(kp_2d[:,:2], 224, inv=True)

    kp_2d = np.hstack([kp_2d, np.ones((kp_2d.shape[0], 1))])

    kp_2d[:,2] = kp_2d[:,2] > 0.3
    kp_2d = np.array(kp_2d, dtype=int)

    rcolor = [255,0,0]
    pcolor = [0,255,0]
    lcolor = [0,0,255]

    skeleton = eval(f'get_{dataset}_skeleton')()
    joint_names = eval(f'get_{dataset}_joint_names')()

    if j_error is not None:
        cv2.putText(
            image, f'MPJPE: {j_error.mean():.1f}',
            (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0)
        )

    # common_lr = [0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]
    for idx, pt in enumerate(kp_2d):
        # if pt[2] > 0: # if visible
        cv2.circle(image, (pt[0], pt[1]), 4, pcolor, -1)
        if j_error is not None:
            cv2.putText(image, f'{j_error[idx]:.1f}', (pt[0]+3, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        if j_uncertainty is not None:
            cv2.putText(image, f'{j_uncertainty[idx]:.6f}',
                        (pt[0]-45, pt[1]-3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

        if print_joints:
            cv2.putText(image, f'{idx}-{joint_names[idx]}',
                        (pt[0]+3, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

    for i,(j1,j2) in enumerate(skeleton):
        # if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0: # if visible
        # if dataset == 'common':
        #     color = rcolor if common_lr[i] == 0 else lcolor
        # else:
        color = lcolor if i % 2 == 0 else rcolor
        if kp_2d[j1, 2] > 0 and kp_2d[j2, 2] > 0:
            pt1, pt2 = (kp_2d[j1, 0], kp_2d[j1, 1]), (kp_2d[j2, 0], kp_2d[j2, 1])
            cv2.line(image, pt1=pt1, pt2=pt2, color=color, thickness=thickness)

    image = np.asarray(image, dtype=float) / 255.
    return image
