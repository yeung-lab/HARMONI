import os
import shutil
import sys

import cv2
import numpy as np

from pose import OpenposeDetector

detector = OpenposeDetector('/pasteur/u/zzweng/projects/HARMONI/data/ckpts')

base_path = "/pasteur/data/cmu_panoptic/panoptic-toolbox/scripts"
out_path = "sampled_frames"
if not os.path.exists(out_path):
    os.makedirs(out_path)

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

videos = [
    '160401_ian1',
    '160401_ian2',
    '160401_ian3',
    '170915_toddler3',
    '170915_toddler4'
]
num_cameras = 31
all_image_paths = []
result_image_paths = []
max_try = 20

for video in videos:
    np.random.seed(28)  # 42, 2024, 123, 28

    for camera in range(num_cameras):
        image_dir = os.path.join(base_path, video, 'hdImgs', f'00_{camera:02d}')

        image_names = os.listdir(image_dir)

        try_i = 0
        while try_i < max_try:  # keep sampling until we get a good one
            sampeld_image_name = np.random.choice(image_names)
            sampled_image_path = os.path.join(image_dir, sampeld_image_name)
            try_i += 1
            oriImg = cv2.imread(sampled_image_path)  # B,G,R order
            canvas, candidate_subset = detector(oriImg, False, draw=True)
            candidate = np.array(candidate_subset['candidate'])
            subset = np.array(candidate_subset['subset'])
            
            if len(subset) < 1: 
                print('Not enough people detected.')
                continue
            
            # cv2.imwrite(f'openpose/{sampeld_image_name}.jpg', canvas)

            image_height, image_width = oriImg.shape[:2]
            kp_person = np.stack([convert_to_op25(candidate, subset[person_i]) for person_i in range(len(subset))])  # (N, 25, 3)
            # choose the bigger person
            person_heights = np.stack([np.max(kp_person[person_i,:,1]) - np.min(kp_person[person_i,:,1]) for person_i in range(len(subset))])
            kp_person1 = kp_person[np.argmax(person_heights)]
            kp_person1 = kp_person1[kp_person1[:,2] > 0.5]

            # check whether visible keypoints are enough.
            min_kps = 6
            if kp_person1.shape[0] < min_kps:
                print('Person not enough keypoints.')
                continue

            # check whether the person is too small.
            min_height = 100
            if np.max(kp_person1[:,1]) - np.min(kp_person1[:,1]) < min_height:
                print('Person too small.')
                continue

            # check whether the person is too close to the edge.
            min_edge = 100
            if np.min(kp_person1[:,0]) < min_edge or np.max(kp_person1[:,0]) > image_width - min_edge:
                print('Person too close to the horizontal edge.')
                continue
            if np.min(kp_person1[:,1]) < min_edge or np.max(kp_person1[:,1]) > image_height - min_edge:
                print('Person too close to the vertical edge.')
                continue

            # import ipdb; ipdb.set_trace()
            all_image_paths.append(sampled_image_path)
            print('Found a good image.')
            break
            
        # result_image_paths.append(os.path.join('../../results/cmu_panoptic_smpla', video + f'_hd_00_{camera:02d}', 'render', sampeld_image_name))


print(all_image_paths)

# sys.exit(0)


for image_path in all_image_paths:
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    video = image_path.split('/')[-4]
    image_name = f'{video}_{image_name}'
    out_image_path = os.path.join(out_path, image_name+'.jpg')
    print('Copying', image_path, 'to', out_image_path)

    shutil.copy(image_path, out_image_path)