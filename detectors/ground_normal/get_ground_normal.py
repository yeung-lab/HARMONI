"""
Sample a few images per shot, and compute ground normal. 

Usage:
python detectors/ground_normal/get_ground_normal.py
"""
import os, sys

sys.path.insert(1, './detectors/panoptic_deeplab')

from loguru import logger
from detectron2.data.detection_utils import read_image
from detectron2.config import get_cfg
from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa

import numpy as np
import pickle
from PIL import Image
import torch
import matplotlib.pyplot as plt

from detectors.panoptic_deeplab.tools_d2.d2.predictor import VisualizationDemo
from detectors.ground_normal.utils import estimate_plane_xy_diff_range, write_depth, init_network
from detectors.ground_normal.utils import read_image as local_read_image
# from detectors.midas.api import MiDaSInference, load_midas_transform

def setup_cfg(config_file, opts, confidence_threshold):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    return cfg


def get_prediction(img, img_input, model, optimize, device):
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize == True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
                .squeeze()
                .cpu()
                .numpy()
        )
    return prediction


def sample_frames_per_scene(shot_info, num_per_scene=5):
    scene_rng = list(zip(shot_info['Start Frame'], shot_info['End Frame']))
    for scene_id in range(len(scene_rng)):
        num_samples = min(num_per_scene, scene_rng[scene_id][1] - scene_rng[scene_id][0])
        frames_sampled = np.random.choice(range(scene_rng[scene_id][0], scene_rng[scene_id][1]), num_samples)
        yield frames_sampled, scene_rng[scene_id]


def compute_ground_normal(predicted_disparity,
                        road, h, w,
                        result_dir='',
                        plane_scale=1.0,
                        negative_plane=False,
                        w_sc=1.0,
                        debug=False
                        ):
    """ Adapted from function get_relative_nomalized_plane_v3
    """
    # get the 2D coordinates of ground
    w_range = np.arange(0, w)
    h_range = np.arange(0, h)
    w_range = np.repeat(w_range, [h]).reshape([w, h])
    w_range = w_range.T
    h_range = np.repeat(h_range, [w]).reshape([h, w])
    x_coords = w_range[road]
    y_coords = h_range[road]
    assert len(x_coords) == len(y_coords)

    if negative_plane is False:
        scale = -1
    else:
        scale = 1
    t = 0

    # un poco mas eficiente
    predicted_disparity_norm = 2*(predicted_disparity[road] / 65535) - 1
    scaled_disparity_road = scale * predicted_disparity_norm + t

    # mas eficiente
    x_world_norm = (2 * w_range[road] / w) - 1
    y_world_norm = (2 * h_range[road] / h) - 1

    # w_range_norm_sc = 2 * w_range_norm
    q_3d = np.stack([w_sc * x_world_norm, y_world_norm, plane_scale * scaled_disparity_road], 1)

    # estimate the plane
    lim = 2
    xrange = [-2*lim, 2*lim]
    yrange = [-lim, lim]

    out_plane_n = os.path.join(result_dir, 'estimated_plane_normalized.ply')
    normal_vect = estimate_plane_xy_diff_range(q_3d, xrange=xrange, yrange=yrange,
                                               name=out_plane_n,
                                               return_normal=True,
                                               debug=debug
                                               )

    return normal_vect


def sample_frames_per_scene(shot_info, num_per_scene=5):
    scene_rng = list(zip(shot_info['Start Frame'], shot_info['End Frame']))
    for scene_id in range(len(scene_rng)):
        frames_sampled = np.random.choice(range(scene_rng[scene_id][0], scene_rng[scene_id][1]), num_per_scene)
        yield frames_sampled, scene_rng[scene_id]


def compute(image_list, output_path, shot_info, num_per_scene):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize panoptic deeplab
    confidence_threshold = 0.5
    opts = ["MODEL.WEIGHTS", "./data/ckpts/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.pth"]
    config_file = "./detectors/panoptic_deeplab/tools_d2/configs/COCO-PanopticSegmentation/panoptic_deeplab_H_48_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml"
    cfg = setup_cfg(config_file, opts, confidence_threshold)
    demo = VisualizationDemo(cfg)
    logger.info("Initialized Panoptic-DeepLab successfully.")

    # initialize MiDaS
    model_path = "./data/ckpts/dpt_large-midas-2f21e586.pt"
    depth_model, _, _, _, _, dpt_transform = init_network('dpt_large', model_path, device, optimize=False)
    logger.info("Initialized MiDaS successfully.")
    
    ground_normal_all_scenes = {}
    for scene_id, (frames, scene_rng) in enumerate(sample_frames_per_scene(shot_info, num_per_scene)):
        normal_vectors = []
        for frame in frames:
            img_name = image_list[frame]
            frame_name = os.path.splitext(os.path.basename(img_name))[0]

            img = local_read_image(img_name)
            dpt_input = dpt_transform({"image": img})["image"]
            # Run Depth Estimation
            prediction = get_prediction(img, dpt_input, depth_model, False, device)
            image = Image.open(img_name)
            w, h = image.size
            image = np.array(image)
            image = image[..., :3]
            # output
            os.makedirs(output_path, exist_ok=True)
            filename = os.path.join(output_path, frame_name)
            out_depth_16bits = write_depth(filename, prediction, bits=2)

            # using this output, its normalized btw [0, 65k]
            out_depth = out_depth_16bits
            zero_idxs = np.where(out_depth == 0.0)
            mean_disp = out_depth.mean()
            out_depth[zero_idxs] = mean_disp
            predicted_disparity = out_depth

            # Run Panoptic Segmentation
            img = read_image(img_name, format="BGR")
            predictions, visualized_output = demo.run_on_image(img)
            pan = predictions['panoptic_seg'][0].cpu().numpy()
            pan = (pan / 1000).astype(int)
            labels = np.unique(pan)
            classes = []

            for l in labels:
                classes.append(demo.metadata.stuff_classes[l])
            # possible classes used as plane
            road = pan == 100
            grass = pan == 125
            rug_merged = pan == 132
            sand = pan == 102
            playingfield = pan == 97
            snow = pan == 105
            floor_other_merged = pan == 122
            dirt_merged = pan == 126
            pavement_merged = pan == 123
            floor_wood = pan == 87
            persons_mask = pan == 0

            road = road | grass | rug_merged | sand | playingfield | snow | \
                floor_other_merged | dirt_merged | pavement_merged | floor_wood

            # out_filename = os.path.join(output_path, f'scene_{scene_id}_{frame_name}_panoptic.png')
            # visualized_output.save(f'scene_{scene_id}_{frame_name}_panoptic.png')
            # img_out = os.path.join(output_path, f'scene_{scene_id}_{frame_name}.jpg')
            # plt.imsave(img_out, image)
            img_out = os.path.join(output_path, f'scene_{scene_id}_{frame_name}_depth_map.jpg')
            out_depth_3 = np.stack([out_depth, out_depth, out_depth], 2) / out_depth.max()
            plt.imsave(img_out, out_depth_3)
            # img_out = os.path.join(output_path, f'scene_{scene_id}_{frame_name}_depth_ground.jpg')
            # plt.imsave(img_out, out_depth_3 * road[..., None])

            # concatenate image, out_depth_3
            image = Image.fromarray(image)
            depth = Image.fromarray((out_depth_3 * 255).astype(np.uint8))
            depth_ground = Image.fromarray((out_depth_3 * road[..., None] * 255).astype(np.uint8))
            
            images = [image, depth, depth_ground]
            # merge image, depth, depth_ground
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height))
            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]
            new_im.save(os.path.join(output_path, f'scene_{scene_id}_{frame_name}.jpg'))

            road_sum = road.sum()
            if road_sum==0:
                print(f'No road detected!! {img_name}')
                continue
            normal_vect = compute_ground_normal(predicted_disparity,
                        road, h, w,
                        result_dir='',
                        plane_scale=1.0,
                        negative_plane=False,
                        w_sc=1.0,
                        debug=False)
            normal_vectors.append(normal_vect)

        # Average over the normal vectors in this scene.
        # first, make the vectors point in the same direction. 
        if len(normal_vectors) == 0: continue
        vec0 = normal_vectors[0]
        normal_vectors_same_dir = [vec0]
        for vec in normal_vectors[1:]:
            if np.dot(vec0, vec) < 0:
                normal_vectors_same_dir.append( - vec)
            else:
                normal_vectors_same_dir.append(vec)
            # print('Cosine similarity:', np.dot(vec0, vec) / np.linalg.norm(vec0) / np.linalg.norm(vec))
        # print(normal_vectors_same_dir)

        mean_normal = np.mean(normal_vectors_same_dir, 0)
        ground_normal_all_scenes[scene_id] = (scene_rng, mean_normal)
        save_normal(mean_normal, output_path)

    with open(os.path.join(output_path, 'ground_normals.pkl'), 'wb') as f:
        pickle.dump(ground_normal_all_scenes, f)
        


def save_normal(normal_vec, out_folder):
    import open3d as o3d
    starting_point = np.array([0, 0, 0])
    sampled_points = np.linspace(starting_point, starting_point + normal_vec)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points)
    o3d.io.write_point_cloud(os.path.join(out_folder, "estimated_normal.ply"), pcd)
    

if __name__ == '__main__':
    
    img_base = './data/demo/vid2'
    output_path = './results/ground_normal'
    os.makedirs(output_path, exist_ok=True)

    ext_list = ['.jpg', '.png', '.jpeg']
    img_list = [os.path.join(img_base, file) for file in os.listdir(img_base) if os.path.splitext(file)[1] in ext_list]
    shot_info = {
        'Start Frame': [0],
        'End Frame': [len(img_list)],
    }
    logger.info(f"Found {len(img_list)} images from {len(shot_info['Start Frame'])} scenes.")
    num_per_scene = 5
    compute(img_list, output_path, shot_info, num_per_scene)
