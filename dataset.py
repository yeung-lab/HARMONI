import os, sys, json
from glob import glob
from collections import defaultdict, Counter

from loguru import logger
import cv2
from torch.utils import data

import pickle
import numpy as np

from detectors.classifier import BodyTypeClassifier
from detectors.openpose import run_on_images as run_openpose

from utils.img_utils import cropout_openpose_one_third, default_img_transform, crop
from detectors.ground_normal.get_ground_normal import compute

class Dataset(data.Dataset):
    """ Holds images, 2D keypoints, their predicted body types (infant vs. adult) and additional info.
    """ 
    def __init__(self, img_folder, out_folder, tracker_type, cfg):
        self.img_folder = img_folder
        self.out_folder = out_folder
        img_paths = sorted(glob(os.path.join(img_folder, '*')))
        ext_list = ['png', 'jpg', 'jpeg']
        self.img_paths = [img_path for img_path in img_paths if img_path.split('.')[-1] in ext_list]
        self.img_names = [os.path.basename(img_path) for img_path in self.img_paths]
        logger.info(f'Found {len(self.img_paths)} images in {img_folder}')
        image = cv2.imread(self.img_paths[0])
        image_height, image_width, _ = image.shape
        self.camera_center = (image_width / 2, image_height / 2)
        logger.info(f'Camera center: {self.camera_center}')

        os.makedirs(os.path.join(out_folder, 'openpose'), exist_ok=True)
        results = run_openpose(self.img_paths, hand=False, vis_path=os.path.join(out_folder, 'openpose'))
        
        num_persons = 0
        person_to_img = {}  # {pid: [img_name, person_index]}
        person_to_det = {}
        for img_path, keypoints_list in results.items():
            img_id = os.path.basename(img_path)

            for i in range(len(keypoints_list)):
                person_to_img[num_persons] = [img_id, i]
                person_to_det[num_persons] = {
                    'img_name': img_id,
                    'keypoints_25': keypoints_list[i],
                    # 'bbox':  # x,y,w,h
                }
                num_persons += 1

        track_to_id, id_to_track, track_id_to_detections = self.run_tracking(person_to_det, tracker_type=tracker_type)
        body_type_classifier = BodyTypeClassifier(cfg.body_type_classifier_path)
        track_body_types = body_type_classifier(track_id_to_detections, self.img_folder, out_folder, classifier=None)
        
        self.num_persons = num_persons
        self.num_ghost_detections = 0
        self.person_to_img = person_to_img  # mapping from person id to img name and person index within the image
        self.person_to_det = person_to_det
        self.track_to_id = track_to_id  # mapping from track id to list of person ids. Used in main fitting loop.
        self.id_to_track = id_to_track
        self.track_body_types = track_body_types
        
        self.transform = default_img_transform()

        self.track_body_types = track_body_types
        print(track_body_types)
        self.print_info()

    def run_openpose(self):
        from detectors.openpose import OpenPoseDetector

    def run_tracking(self, person_to_det, tracker_type):
        if tracker_type == 'dummy':
            # dummy tracker. Just assign each person to a unique track id.
            id_to_track = {}
            track_to_id = defaultdict(list)
            track_id_to_detections = defaultdict(list)
            for person_id in range(len(person_to_det)):
                track_id = person_id
                id_to_track[person_id] = person_id
                track_to_id[track_id].append(person_id)
                track_id_to_detections[track_id].append(person_to_det[person_id])

        elif tracker_type == 'phalp':
            self.run_phalp() # TODO: implement this
        else:
            raise NotImplementedError(f'Unknown tracker type: {tracker_type}')

        return track_to_id, id_to_track, track_id_to_detections

    def process_phalp(self):
        # TODO (Jen): add code for runnning PHALP somewhere.
        person_to_detection_map = {}
        
        suffix = '_openpose_transreid_max_age_10_ghost_head_fast_may'

        phalp_detections = os.path.join(out_folder, "../phalp_detection{}.pickle".format(suffix))
        phalp_results = os.path.join(out_folder, "../phalp_results{}.pickle".format(suffix))
        phalp_visuals = os.path.join(out_folder, "../phalp_visuals{}.pickle".format(suffix))

        with open(phalp_detections, "rb") as f:
            detections = pickle.load(f)
        with open(phalp_results, "rb") as f:
            tracking_results = pickle.load(f)
        with open(phalp_visuals, "rb") as f:
            visuals = pickle.load(f)  # this contains all ghost detections
        print(len(tracking_results), len(detections), len(visuals))
        self.handle_ghost_detections(tracking_results, detections, visuals)


        # match tracking results to detections by bounding box
        track_id_to_detections = defaultdict(list)
        num_ghost_detections = 0

        detections_by_frame = defaultdict(list)
        for frame_id in range(len(detections)):
            for det in detections[frame_id]:
                detections_by_frame[det["img_name"]].append(det)
        
        frame_list = sorted(list(tracking_results.keys()))
        # track_body_types = {}
        adult_counts_per_image = {}
        infant_counts_per_image = {}
        for _, frame_name in enumerate(frame_list):  # 'frame000089.jpg'
            # frame_id = int(frame_name.split('.')[0].split('frame')[-1])
            tracking_res = tracking_results[frame_name]  # e.g. [[8, 9], [[153.0, 5.0, 484.0, 763.0], [711.0, 1.0, 335.0, 839.0]], 89]
            det_res = detections_by_frame[frame_name]  # e.g. [{'bbox': [153, 5, 484, 763], 'time': 89, 'img_name': 'frame000089.jpg', 'det_index': 0, 'keypoints_25': ..},..]
            # assert (len(det_res) ==0 or det_res[0]["img_name"] == frame_name)
            adult_counts_per_image[frame_name] = sum([det.get("class_id", "")=="adult" for det in det_res])
            infant_counts_per_image[frame_name] = sum([det.get("class_id", "")=="infant" for det in det_res])
            for track_id, track_box in zip(tracking_res[0], tracking_res[1]):
                for det_id, det in enumerate(det_res):
                    if np.array_equal(det["bbox"], track_box):
                        track_id_to_detections[track_id].append(det)
                        track_to_id[track_id].append(num_persons)
                        id_to_track[num_persons] = track_id
                        person_to_img_map[num_persons] = (frame_name, det["det_index"])
                        person_to_detection_map[num_persons] = det
                        num_persons += 1
                        # if track_id in track_body_types.keys() or "ghost" in det.keys():
                        #     continue
                        # else:
                        #     track_body_types[track_id] = (det["class_id"], None)
                            
                       
                        if "ghost" in det.keys():
                            num_ghost_detections += 1

        logger.info('{} persons from {} tracks detected in {} images.'.format(num_persons, len(track_to_id), len(frame_list)))
        logger.info("{} of them are ghost detections from PHALP.".format(num_ghost_detections))
        self.person_to_detection_map = person_to_detection_map
        self.num_ghost_detections = num_ghost_detections

    def get_sorted_track_by_body_type(self):
        tracks = self.track_body_types.keys()
        adult_tracks, infant_tracks = [], []
        for track_id in tracks:
            if self.track_body_types[track_id][0] == 'adult':
                adult_tracks.append(track_id)
            else:
                infant_tracks.append(track_id)
        return sorted(adult_tracks), sorted(infant_tracks)
    
    def print_info(self):
        logger.info("{} persons from {} tracks detected in {} images.".format(self.num_persons, len(self.track_to_id), len(self.img_paths)))
        logger.info("{} persons are ghost detections from PHALP.".format(self.num_ghost_detections))

        adult_tracks, infant_tracks = self.get_sorted_track_by_body_type()
        adult_counts = [len(self.track_to_id[track_id]) for track_id in adult_tracks]
        infant_counts = [len(self.track_to_id[track_id]) for track_id in infant_tracks]
        logger.info("{} tracks are infant, {} tracks are adult.".format(len(infant_tracks), len(adult_tracks)))
        logger.info("{} persons are infant, {} persons are adult.".format(sum(infant_counts), sum(adult_counts)))

    def handle_ghost_detections(self, results, detections, visuals):
        # For each track, find the last time (t_end) it appears in results[t_end][0]. 
        # Keep the ghost detections (visuals[*][6] > 0) between [t_start, t_end], and insert in detections and results.
        # (i.e. Dispose all ghost detections after t_end)
        track_last_appear = dict()
        for frame_name in np.sort(list(results.keys())):
            res = results[frame_name]
            for track_id in res[0]:
                track_last_appear[track_id] = frame_name

        def fname_cmp(fname1, fname2):
            # returns 1 iff frame id 1 <= frame id 2
            fid1, fid2 = int(fname1[5:-4]), int(fname2[5:-4])
            return fid1 <= fid2

        for frame_id, frame_name in enumerate(np.sort(list(visuals.keys()))):
            visual_res = visuals[frame_name]
            track_ids, bbox = visual_res[0], visual_res[1]
            is_ghost = np.array(visual_res[6]) > 0
            if len(is_ghost) != len(track_ids): continue
            for i_, track_id in enumerate(track_ids):
                if is_ghost[i_] and fname_cmp(frame_name, track_last_appear[track_id]):
                        det = {
                            "det_index": len(detections[frame_id]), 
                            "bbox": bbox[i_],
                            "keypoints_25": np.zeros((25, 3)), 
                            "img_name": frame_name,
                            "ghost": True
                        }
                        detections[frame_id].append(det)
                        results[frame_name][0].append(track_id)
                        results[frame_name][1].append(bbox[i_])

    def __len__(self):
        return self.num_persons

    def __getitem__(self, idx):
        img_name, p_i = self.person_to_img[idx]
        try:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, img_name)), cv2.COLOR_BGR2RGB)
        except:
            print('could not find image at: ', os.path.join(self.img_folder, img_name))
            sys.exit(0)

        detection = self.person_to_det[idx]
        # kp_118 = np.zeros((118, 3))
        kp = detection.get("keypoints_25", np.zeros((25, 3)))  # (25, 3)
        # kp_118[:25] = kp
        is_ghost = 'ghost' in detection.keys()

        if kp[:, 2].sum() > 0:
            crop_out = cropout_openpose_one_third(img, kp)
            cropped_img = crop_out['cropped_image']  # cropped image
            yhxw = np.array((crop_out['crop_boundary']['y'], crop_out['crop_boundary']['h'], 
                    crop_out['crop_boundary']['x'], crop_out['crop_boundary']['w']))
        else:
            box_detectron2 = detection["bbox"]  # (x,y,w,h)
            x,y,w,h = box_detectron2
            center_x = (x+w/2)
            center_y = (y+h/2)
            side = max(w, h)
            x_min, y_min, x_max, y_max = [center_x-side/2, center_y-side/2, center_x+side/2, center_y+side/2]
            center = [center_x, center_y]
            scale = max(x_max-x_min, y_max-y_min) * 1.2
            cropped_img = crop(img, center, scale, [224,224], edge_padding=True).astype(np.uint8)
            yhxw = np.array((y_min.item(),side.item(),x_min.item(),side.item()))

        track_id = self.id_to_track[idx]
        body_type = self.track_body_types[track_id][0]
        result = {'img_name': img_name, 'keypoints': kp, 'body_type': body_type, 
                  'yhxw': yhxw, 'idx': idx, 'norm_cropped_img': self.transform(cropped_img), 'is_ghost': is_ghost}
        return result

    def estimate_ground_plane_normal(self):
        logger.info("Estimating ground plane normal...")
        output_path = os.path.join(self.out_folder, 'ground_normal')
        os.makedirs(output_path, exist_ok=True)
        shot_info = {
            'Start Frame': [0],
            'End Frame': [len(self.img_paths)],
        }  # TODO (Jen): dummy shot info. Integrate real one later.
        num_per_scene = 5
        ground_normal = compute(self.img_paths, output_path, shot_info, num_per_scene)
        self.ground_normals = ground_normal  # scene_id: (scene_rng, mean_normal)
        logger.info("Ground plane normal estimated.")