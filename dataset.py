import os, sys, json
from glob import glob
from collections import defaultdict

import cv2
from loguru import logger
from torch.utils import data
import numpy as np
import omegaconf


from detectors.ground_normal.get_ground_normal import compute
from utils.img_utils import cropout_openpose_one_third, default_img_transform, crop


class Dataset(data.Dataset):
    """ Holds images, 2D keypoints, their predicted body types (infant vs. adult) and additional info.

    Pipeline 1: run openpose, run tracker, classify bbox.
    Pipeline 2: grounded dino, run tracker.
    """ 
    def __init__(self, img_folder, out_folder, tracker_type, pipeline, cfg):
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

        if pipeline == 1:
            from detectors.classifier import BodyTypeClassifier
            from detectors.openpose import run_on_images as run_openpose

            os.makedirs(os.path.join(out_folder, 'openpose'), exist_ok=True)
            results = run_openpose(self.img_paths, hand=False, 
                                #    vis_path=os.path.join(out_folder, 'openpose')
                                )
            
            num_persons = 0
            person_to_img = {}  # {pid: [img_name, person_index]}
            img_to_img_id = {}
            person_to_det = {}
            for img_id, (img_path, keypoints_list) in enumerate(results.items()):
                img_name = os.path.basename(img_path)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img_to_img_id[img_name] = img_id
                
                for i in range(len(keypoints_list)):
                    person_to_img[num_persons] = [img_name, i]
                    kp = keypoints_list[i]
                    crop_out = cropout_openpose_one_third(img, kp)
                    xywh = [crop_out['crop_boundary']['x'], crop_out['crop_boundary']['y'],
                            crop_out['crop_boundary']['w'], crop_out['crop_boundary']['h']]
                    person_to_det[num_persons] = {
                        'img_name': img_name,
                        'keypoints_25': kp,
                        'bbox': xywh
                    }
                    num_persons += 1
                    
            body_type_classifier = BodyTypeClassifier(cfg.body_type_classifier_path)
            track_to_id, id_to_track, track_id_to_detections = self.run_tracking(person_to_det, tracker_type=tracker_type)
            track_body_types = body_type_classifier(track_id_to_detections, self.img_folder, out_folder, classifier=None)

        elif pipeline == 2:
                
            # TODO: sort out env conflict and use run_on_images instead.
            logger.info('Running Grounded Dino on images in {}'.format(img_folder))
            results_json = os.path.join(os.path.dirname(img_folder), 'ground_dino.json')
            os.system('/pasteur/u/zywang/env/conda/envs/3d/bin/python detectors/grounded_dino/api.py --input_video {} --output_file {}'.format(
                img_folder, results_json))
            with open(results_json, 'r') as f:
                detection_list = json.load(f)

            # from detectors.grounded_dino.api import run_on_images_cmd as run_grounded_dino
            # detection_list = run_grounded_dino(img_folder)
            num_persons = 0
            person_to_img = {}  # {pid: [img_name, person_index]}
            img_to_img_id = {}
            person_to_det = {}
            id_to_track = {}
            # dummy tracking
            baby_track_id = 1
            adult_track_id = 0
            track_to_id = defaultdict(list)
            track_id_to_detections = defaultdict(list)

            for img_id, detections in enumerate(detection_list):
                img_name = os.path.basename(detections[0][0])
                img_to_img_id[img_name] = img_id
                
                for i, det in enumerate(detections):
                    person_to_img[num_persons] = [img_name, i]
                    xyxy = det[1]  # x,y,x,y
                    xywh = [xyxy[0], xyxy[1], xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
                    person_to_det[num_persons] = {
                        'img_name': img_name,
                        'bbox': xywh,
                        # 'keypoints_25': np.zeros((25, 3)),
                    }
                    
                    if det[3] == 'infant':
                        track_id_to_detections[baby_track_id].append(person_to_det[num_persons])
                        track_to_id[baby_track_id].append(num_persons)
                        id_to_track[num_persons] = baby_track_id
                    else:
                        track_id_to_detections[adult_track_id].append(person_to_det[num_persons])
                        track_to_id[adult_track_id].append(num_persons)
                        id_to_track[num_persons] = adult_track_id

                    num_persons += 1
            
            track_body_types = {0: ('adult', None), 1: ('infant', None)}
            self.visualize_tracks(os.path.join(out_folder, 'tracks'), track_id_to_detections)
        else:
            raise NotImplementedError(f'Unknown pipeline type: {cfg.pipeline}')
            
        self.num_persons = num_persons
        self.num_ghost_detections = 0
        self.person_to_img = person_to_img  # mapping from person id to img name and person index within the image
        self.img_to_img_id = img_to_img_id
        self.person_to_det = person_to_det
        self.transform = default_img_transform()
        self.track_to_id = track_to_id  # mapping from track id to list of person ids. Used in main fitting loop.
        self.id_to_track = id_to_track
        self.track_body_types = track_body_types
        
        self.print_info()

    def visualize_tracks(self, out_folder, track_id_to_detections, num_per_track=5):
        os.makedirs(out_folder, exist_ok=True)
        for track_id, detections in track_id_to_detections.items():
            crops = []
            detections_to_visualize = np.random.choice(detections, min(num_per_track, len(detections)), replace=False)
            for det in detections_to_visualize:
                img_name = det['img_name']
                img = cv2.imread(os.path.join(self.img_folder, img_name))
                bbox = det['bbox']
                x,y,w,h = bbox
                cropped_img = crop(img, [x+w/2, y+h/2], max(w, h), [224,224], edge_padding=True).astype(np.uint8)
                crops.append(cv2.resize(cropped_img, (224, 224)))

            crops = np.concatenate(crops, axis=1)
            cv2.imwrite(os.path.join(out_folder, f'track_{track_id}.jpg'), crops)

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
            sys.path.append(os.path.abspath("./trackers"))
            from trackers.phalp import PHALP
            cfg = omegaconf.OmegaConf.load(os.path.abspath("./trackers/phalp_config.yaml"))
            cfg.video.output_dir = self.out_folder
            phalp_tracker = PHALP(cfg)
            track_to_id, id_to_track = phalp_tracker.track(self)
            track_id_to_detections = defaultdict(list)
            for track_id, person_ids in track_to_id.items():
                for person_id in person_ids:
                    track_id_to_detections[track_id].append(person_to_det[person_id])
        else:
            raise NotImplementedError(f'Unknown tracker type: {tracker_type}')

        return track_to_id, id_to_track, track_id_to_detections

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

        # if kp[:, 2].sum() > 0:
        #     crop_out = cropout_openpose_one_third(img, kp)
        #     cropped_img = crop_out['cropped_image']  # cropped image
        #     yhxw = np.array((crop_out['crop_boundary']['y'], crop_out['crop_boundary']['h'], 
        #             crop_out['crop_boundary']['x'], crop_out['crop_boundary']['w']))
        # else:
        x,y,w,h = detection["bbox"]  # (x,y,w,h)
        center_x = (x+w/2)
        center_y = (y+h/2)
        side = max(w, h)
        x_min, y_min, x_max, y_max = [center_x-side/2, center_y-side/2, center_x+side/2, center_y+side/2]
        center = [center_x, center_y]
        scale = max(x_max-x_min, y_max-y_min) * 1.2
        cropped_img = crop(img, center, scale, [224,224], edge_padding=True).astype(np.uint8)
        yhxw = np.array((y_min, side, x_min, side))

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