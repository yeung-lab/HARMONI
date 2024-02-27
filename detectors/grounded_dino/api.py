import os
import json
import glob
import argparse

import torch
from loguru import logger
from tqdm import tqdm
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, load_image, predict



def run_on_images(image_paths, args=None):

    if args is not None:
        CONFIG_PATH = args.config_path
        CHECKPOINT_PATH = args.checkpoint_path
        DEVICE = args.device
        TEXT_PROMPT = args.text_prompt
        BOX_TRESHOLD = args.box_threshold
        TEXT_TRESHOLD = args.text_threshold
    else:
        CONFIG_PATH = 'config/GroundingDINO_SwinT_OGC.py'
        CHECKPOINT_PATH = '/pasteur/u/zywang/projects/offshelf/grounded_segment_anything/weights/groundingdino_swint_ogc.pth'
        DEVICE = 'cuda'
        TEXT_PROMPT = 'Parent. Baby.'
        BOX_TRESHOLD = 0.35
        TEXT_TRESHOLD = 0.25

    model = load_model(CONFIG_PATH, CHECKPOINT_PATH)

    detection_list = []

    for img in tqdm(image_paths):
        image_source, image = load_image(img)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=DEVICE,
        )

        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        mapped_phrases = [phrase_to_type(phrase) for phrase in phrases]

        detection = list(zip([img]*len(boxes), boxes.tolist(), logits.tolist(), mapped_phrases))
        detection.sort(key=lambda x: x[-1]) # put baby first
        detection_list.append(detection)

    return detection_list


def phrase_to_type(phrase):
    adult_phrases = ['adult', 'parent', 'human']
    baby_phrases = ['baby', 'child']
    if any([p in phrase.lower() for p in baby_phrases]):
        return 'infant'
    elif any([p in phrase.lower() for p in adult_phrases]):
        return 'adult'
    else:
        raise ValueError('Unknown phrase: {}'.format(phrase))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', default=str)
    parser.add_argument('--output_file', type=str, default='ground_dino.json')
    parser.add_argument('--config_path', type=str, default='./detectors/grounded_dino/config/GroundingDINO_SwinT_OGC.py')
    parser.add_argument('--checkpoint_path', type=str, default='/pasteur/u/zywang/projects/offshelf/grounded_segment_anything/weights/groundingdino_swint_ogc.pth')
    parser.add_argument('--box_threshold', type=float, default=0.35)
    parser.add_argument('--text_threshold', type=float, default=0.25)
    parser.add_argument('--text_prompt', type=str, default='Parent. Adult. Baby. Child.')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    video_path = args.input_video
    img_path = glob.glob(os.path.join(video_path, '*.png'))
    img_path += glob.glob(os.path.join(video_path, '*.jpg'))

    detections_list = run_on_images(img_path, args)
    with open(args.output_file, 'w') as f:
        json.dump(detections_list, f)
