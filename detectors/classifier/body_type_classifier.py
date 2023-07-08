import os
import shutil
from collections import Counter

import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
import torch.nn as nn
import torchvision

from utils.img_utils import cropout_openpose_torso, torso_area, default_img_transform, unnormalize


def load_semi_sup_classifier(model_path):
    model = torchvision.models.resnet101()
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 2)
    state_dict = torch.load(model_path, map_location='cpu')
    logger.info('Loaded pretained body type classifier from '+ model_path)
    model.load_state_dict(state_dict['ema_state_dict'])
    model.eval()
    label_dict = {'infant': 1, 'adult': 0}
    return model, label_dict


class BodyTypeClassifier():
    def __init__(self, model_path, device='cuda') -> None:
        self.classifier, self.label_dict = load_semi_sup_classifier(model_path)
        self.device = device
        self.classifier.to(self.device)
        self.img_transform = default_img_transform(input_pil=True, size=256)
        self.infant_thres = 0.6 # if prob for infant is > thres, then it is an infant. 

    def __call__(self, track_id_to_detections, img_dir, out_folder, classifier=None,
                save_samples=True, sample_num=20):
        if save_samples:
            sample_folder = os.path.join(out_folder, 'sampled_tracks')
            os.makedirs(sample_folder, exist_ok=True)
        
        track_id_to_type = {}
        for track_id, detections in tqdm(track_id_to_detections.items()):
            if len(detections) <= sample_num:
                sampled_idxs = range(len(detections))
            else:
                sampled_idxs = sorted(np.random.choice(range(len(detections)), sample_num, replace=False))
        
            # use the sampled detections to classify body type
            all_cropped = []
            for i in sampled_idxs:
                img_path = os.path.join(img_dir, detections[i]["img_name"])
                img = Image.open(img_path)
                keypoints = detections[i]["keypoints_25"]
                if keypoints[:, 2].sum() > 0:
                    box_dict = cropout_openpose_torso(
                        cv2.imread(img_path)[:,:,::-1], 
                        keypoints)['crop_boundary']
                    box = [box_dict['x'], box_dict['y'], box_dict['x']+box_dict['w'], box_dict['y']+box_dict['h']]
                else:
                    x,y,w,h = detections[i]["bbox"]  # (x,y,w,h)
                    center_x = (x+w/2)
                    center_y = (y+h/2)
                    side = max(w, h)
                    box = [center_x-side/2, center_y-side/2, center_x+side/2, center_y+side/2]  # (x,y,x2,y2)
                cropped = self.img_transform(img.crop((box)))
                all_cropped.append(cropped)
                
            if self.classifier is not None:
                with torch.no_grad():
                    preds_prob = self.classifier(torch.stack(all_cropped).to(self.device))
                preds_prob = preds_prob.cpu()
                
                preds_prob = torch.softmax(preds_prob, 1)
                preds = preds_prob.argmax(1)
                preds = preds_prob[:, 1] > self.infant_thres 
                pred_counter = Counter(preds.numpy())

                adult_mean_prob = preds_prob[:, self.label_dict['adult']].mean().item()
                infant_mean_prob = preds_prob[:, self.label_dict['infant']].mean().item()
                
                try:
                    mean_torso_area = np.mean([torso_area(t['keypoints_25']) for t in detections])
                except:
                    mean_torso_area = np.mean([t['bbox'][2]*t['bbox'][3] for t in detections])

                if pred_counter[self.label_dict['adult']]/ len(sampled_idxs) >= 0.5: # if majority of the predictions says adult, then it's adult
                    track_id_to_type[track_id] = ['adult', adult_mean_prob, infant_mean_prob, mean_torso_area]
                else:
                    track_id_to_type[track_id] = ['infant', adult_mean_prob, infant_mean_prob, mean_torso_area]

            else:
                class_id = None
                for det in detections:
                    if det.get('class_id') is not None:
                        track_id_to_type[track_id] = (det.get("class_id"), 0, 0, 0)
                        class_id = det.get("class_id")
                        break
                if class_id is None:
                    import pdb; pdb.set_trace()

            if save_samples:
                text = 'Start:' + detections[0]["img_name"] + ', End:' + detections[-1]["img_name"]
                if classifier is not None:
                    text += '\nPredicted:'+track_id_to_type[track_id][0]+'\nPreds:'+str(preds) +'\nSoftmax:'+ str(preds_prob.detach().numpy())
                else:
                    text += '\nTrack id: ' + str(track_id) + '. Predicted:'+track_id_to_type[track_id][0]
                all_cropped = unnormalize(torch.stack(all_cropped))
                pimg = torchvision.transforms.ToPILImage()(torchvision.utils.make_grid(all_cropped, nrow=5))
                # pad the bottom of the image with white space
                pimg = ImageOps.expand(pimg, border=(0, 40, 0, 0), fill='white')
                draw = ImageDraw.Draw(pimg)
                font_type = ImageFont.truetype("data/arial.ttf", 12)
                draw.text((5, 5), text, font=font_type, fill=(0, 0, 0))
                pimg.save(os.path.join(sample_folder, 'track_{}.jpg'.format(track_id)))  

        return track_id_to_type
