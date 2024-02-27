import os
import json

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


data_path = '/pasteur/data/cmu_panoptic/panoptic-toolbox/scripts/'

seq_names = ['170915_toddler5', '160906_ian1', '160906_ian2', '160906_ian3', '160906_ian5']

for seq_name in seq_names:
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

        
    # Choose only HD cameras for visualization
    hd_cam_idx = zip([0] * 30,range(0,30))
    hd_cameras = [cameras[cam].copy() for cam in hd_cam_idx]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # Draw selected camera subset in blue
    for cam_id, cam in enumerate(hd_cameras):
        cc = (-cam['R'].transpose()*cam['t'])
        ax.scatter(cc[0].item(), cc[1].item(), c=[0,0,1], s=10)
        ax.text(cc[0].item(), cc[1].item(), str(cam_id), color=[0,0,1])

    # ax.view_init(elev = -90, azim=-90)
    #ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    ax.axis('equal')

    # save fig
    plt.savefig('camera_poses_{:s}.png'.format(seq_name), dpi=300)
        