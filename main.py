from collections import defaultdict
import os, sys, json
import os.path as osp
import joblib
import yaml

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from loguru import logger
from torchgeometry import rotation_matrix_to_angle_axis

from cmd_parser import parse_config
import constants as cfg
import hps
from dataset import Dataset
from results import Results
from utils.vid_utils import video_to_images, images_to_gif, images_to_mp4
from hps.smpl_utils import get_original, collect_results_for_image_dapa, cam_crop2full
from postprocess.temporal_smplify import TemporalSMPLify
from postprocess.one_euro_filter import OneEuroFilter
from visualization.pyrender import render
from downstream.calc_downstream import get_downstream_labels


def main(args):
    # convert video to images
    if args.video is not None and os.path.exists(args.video):
        images_folder = osp.join(args.out_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        video_to_images(args.video, images_folder, start_frame=args.start_frame, end_frame=args.end_frame, fps=args.fps)
        logger.info('Saved frames to ' + images_folder)

    elif args.images is not None:
        images_folder = args.images
    
    out_folder = args.out_folder

    if args.ground_constraint:
        args.run_smplify = True

    if os.path.exists(out_folder):
        logger.warning('Folder ' + out_folder + ' exists. Overwriting it...')
    else:
        os.makedirs(out_folder, exist_ok=True)

    # save args as config.yaml
    with open(os.path.join(out_folder, 'config.yaml'), 'w') as f:
        yaml.dump(vars(args), f)

    batch_size = args.batch_size
    device = 'cuda'
    
    # build dataset and save it to disk.
    dataset_path = os.path.join(out_folder, 'dataset.pt')
    if args.use_cached_dataset and os.path.exists(dataset_path):  
        logger.info('Loading dataset from ' + dataset_path) 
        dataset = joblib.load(dataset_path)
    else:  
        dataset = Dataset(
            images_folder, out_folder=out_folder, tracker_type=args.tracker_type, pipeline=args.pipeline, cfg=cfg
        )
        if args.ground_constraint:
            dataset.estimate_ground_plane_normal()
        joblib.dump(dataset, dataset_path)
        logger.info('Saved dataset.pt to folder: ' + out_folder)
        if args.dryrun:
            sys.exit(0)

    dataset.print_info()

    track_overwrite = eval(args.track_overwrite)
    if len(track_overwrite) > 0:
        logger.info('Overwriting body types for tracks: ' + str(track_overwrite))
        for track_id, body_type in track_overwrite.items():
            dataset.track_body_types[track_id][0] = body_type
        logger.info('Overwriting done.')
        dataset.print_info()

    results_holder = Results()
    camera_center = dataset.camera_center
    orig_img_width, orig_img_height = camera_center[0] * 2, camera_center[1] * 2
    cam_focal_length = args.camera_focal
    fov = 60
    cam_focal_length = orig_img_width / (2 * np.tan(fov * np.pi / 360))
    logger.info('Camera focal length is set to ' + str(cam_focal_length))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    out_render_path = os.path.join(out_folder, 'render')
    if not os.path.exists(out_render_path):
        os.makedirs(out_render_path)

    if args.run_smplify:
        smplify_runner = TemporalSMPLify(
            step_size=1e-2,
            num_iters=args.smplify_iters,
            focal_length=cam_focal_length,
            use_lbfgs=False,
            device=torch.device('cuda'),
            max_iter=20, 
            ground_weight=args.ground_weight
        )
    else:
        smplify_runner = None
    
    adult_tracks, infant_tracks = dataset.get_sorted_track_by_body_type()
    cam_pitch = cam_roll = 0.  # settings these would make render_rotmat an identity matrix.
    cam_vfov = 0.6  # this does not matter
    cam_params = np.array([cam_vfov, cam_pitch, cam_roll, cam_focal_length])   # Identity matrix
    create_body_pose = True

    if not args.render_only:
        # load pretrained DAPA model
        if args.hps == 'dapa':
            adult_model = hps.get_dapa_model(cfg.dapa_adult_model, cfg.smpl_mean_params).cuda().eval()
            infant_model = hps.get_dapa_model(cfg.dapa_child_model, cfg.smpl_mean_params).cuda().eval()
            
        elif args.hps == 'cliff':
            from hps.cliff import cliff_hr48, strip_prefix_if_present
            cliff_model = cliff_hr48(cfg.smpl_mean_params).cuda()
            # Load the pretrained model
            logger.info("Load the CLIFF checkpoint from path:", cfg.cliff_hr48_model)
            state_dict = torch.load(cfg.cliff_hr48_model)['model']
            state_dict = strip_prefix_if_present(state_dict, prefix="module.")
            cliff_model.load_state_dict(state_dict, strict=True)
            cliff_model.eval()

            # infant_model = hps.get_dapa_model(cfg.dapa_child_model, cfg.smpl_mean_params).cuda().eval()
        else:
            create_body_pose = False  # if use vposer, then create_body_pose = False
            raise NotImplementedError('Unknown hps model: ' + args.hps)

        if args.ground_constraint:
            # Note: fit all tracks first, and then use the mean adult/child ankle location as floor to constrain the humans.
            if args.ground_anchor == 'adult_bottom':
                tracks_to_be_fitted = adult_tracks + adult_tracks + infant_tracks
                num_initial_tracks = len(adult_tracks)
            elif args.ground_anchor == 'child_bottom':
                tracks_to_be_fitted = infant_tracks + infant_tracks + adult_tracks
                num_initial_tracks = len(infant_tracks)

        else:
            tracks_to_be_fitted = adult_tracks + infant_tracks
            num_initial_tracks = len(tracks_to_be_fitted)

        # Begin main loop
        for fit_id, track_id in enumerate(tracks_to_be_fitted):
            refine_with_ground = fit_id >= num_initial_tracks  # iterate over tracks a second time and use the floor
            if fit_id == num_initial_tracks and args.ground_constraint:
                logger.info('Fitting the rest of the tracks with ground constraint.')
                # compute ankles per scene
                scene_to_mean_floor = {}
                for scene_id, ((scene_start, scene_end), normal_vec) in dataset.ground_normals.items():
                    if args.ground_anchor == 'adult_bottom':
                        saved_ankles = results_holder.adult_bottom
                    else:
                        saved_ankles = results_holder.infant_bottom
                    ankle_projections = [
                        np.dot(np.array(saved_ankles[dataset.img_names[frame_id]]), normal_vec)
                        for frame_id in range(scene_start, scene_end) if dataset.img_names[frame_id] in saved_ankles
                    ]
                    
                    if len(ankle_projections) == 0: 
                        scene_to_mean_floor[scene_id] = None
                        continue
                    ankle_projections = np.concatenate(ankle_projections).flatten()
                    mean = np.mean(ankle_projections)
                    std = np.std(ankle_projections)
                    mean_floor = np.mean(ankle_projections[np.abs(ankle_projections - mean) < std])  # filter out the outliers
                    scene_to_mean_floor[scene_id] = mean_floor * normal_vec  # location of the anchor ankle.

                # clear the saved ankles
                results_holder.adult_bottom = defaultdict(list)
                results_holder.infant_bottom = defaultdict(list)
            
            pidxs = dataset.track_to_id[track_id]
            body_type = dataset.track_body_types[track_id][0]
            logger.info('Fitting track {} of length {}. Body type is {}...'.format(track_id, len(pidxs), body_type))

            for batch_start_i in range(0, len(pidxs), batch_size):
                batch_pidxs = pidxs[batch_start_i: batch_start_i+batch_size]
                cur_batch_size = len(batch_pidxs)
                batch = dataloader.collate_fn([dataset[i] for i in batch_pidxs])
                yhxw = batch['yhxw']
                cx, cy = yhxw[:, 2] + yhxw[:, 3] / 2., yhxw[:, 0] + yhxw[:, 1] / 2.
                
                start_image_id = dataset.img_to_img_id[batch['img_name'][0]]

                ####################################
                # initialize pose, shape with DAPA #
                ####################################
                if args.ground_constraint:
                    # load ground normal, and set cam_rotmat to be identity matrix
                    for scene_id, (scene_rng, normal_vec_) in dataset.ground_normals.items():
                        if start_image_id >= scene_rng[0] and start_image_id <= scene_rng[1]:
                            normal_vec = normal_vec_ 
                            break
                
                    if refine_with_ground:
                        # logger.info(f'This scene is from frame {scene_rng[0]} to {scene_rng[1]}. Ground normal vector is {str(normal_vec)}.')
                        ground_y = scene_to_mean_floor[scene_id]
                        ground_y = torch.from_numpy(ground_y).to(device).float()
                        # logger.info('Refine with floor at ' + str(ground_y))
                    else:
                        ground_y = None

                    normal_vec = torch.from_numpy(normal_vec).to(device).float()

                if args.hps == 'dapa':
                    if body_type == 'adult':
                        with torch.no_grad():
                            pred_rotmat, pred_betas, pred_camera = adult_model(batch['norm_cropped_img'].to(device))
                    else:
                        with torch.no_grad():
                            pred_rotmat, pred_betas, pred_camera = infant_model(batch['norm_cropped_img'].to(device))

                elif args.hps == 'cliff':
                    norm_img = batch['norm_cropped_img'].to(device)
                    b = torch.max(yhxw[:, 1], yhxw[:, 3])
                    img_w, img_h = orig_img_width, orig_img_height
                    bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1).to(device).float()
                    # The constants below are used for normalization, and calculated from H36M data.
                    # It should be fine if you use the plain Equation (5) in the paper.
                    bbox_info[:, :2] = bbox_info[:, :2] / cam_focal_length * 2.8  # [-1, 1]
                    bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * cam_focal_length) / (0.06 * cam_focal_length)  # [-1, 1]

                    with torch.no_grad():
                        pred_rotmat, pred_betas, pred_camera = cliff_model(norm_img, bbox_info)
                else:
                    raise NotImplementedError('Unknown hps model: ' + args.hps)
                
                init_betas = pred_betas
                pred_rotmat_hom = torch.cat(
                    [pred_rotmat.view(-1, 3, 3),
                    torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1).expand(cur_batch_size*24, -1, -1)
                    ], dim=-1)
                pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(cur_batch_size, -1)
                pred_pose[torch.isnan(pred_pose)] = 0.0
                init_global_orient = pred_pose[:,:3]
                init_pose = pred_pose[:, 3:]

                focal_length = torch.tensor([cam_focal_length], dtype=torch.float32, device=device).expand(cur_batch_size)
                full_image_shape = torch.tensor([orig_img_height, orig_img_width]).to(device).float().unsqueeze(0).expand(cur_batch_size, -1)
                scale = torch.max(yhxw[:, 1], yhxw[:, 3]).to(device) / 200
                center = torch.stack((cx, cy), dim=-1).to(device)
                init_transl = cam_crop2full(pred_camera, center, scale, full_image_shape, focal_length)

                if args.hps == 'cliff' and body_type == 'infant':
                    print('Scaling down the camera focal length for infants')
                    init_transl[:, 0] *= 0.5  # x axis.
                    init_transl[:, 1] *= 0.3  # y axis.
                    init_transl[:, 2] *= 0.2
  
                ##############################################
                # Collect results and optionally run SMPLify #
                ##############################################
                if not args.run_smplify or (not refine_with_ground):
                    # parse results
                    _, results_batch = collect_results_for_image_dapa(
                        pred_pose, pred_betas, pred_camera, None, batch, cam_focal_length, orig_img_width, orig_img_height,
                        smpl_type=args.smpl_model, kid_age=args.kid_age)
                    # update results in this batch
                    results_holder.update_results(batch['idx'].numpy(), results_batch)
                else:
                    model_type = 'smil' if body_type == 'infant' else 'smpl'

                    init_betas = init_betas[:1, ...]
                    smplify_results, reproj_loss = smplify_runner(
                        model_type, init_global_orient, init_pose, init_betas, init_transl, 
                        camera_center, batch['keypoints'], ground_y, normal_vec)
                    refined_thetas = smplify_results['theta']
                    refined_transl = refined_thetas[:, :3]
                    refined_pose = refined_thetas[:, 3:-10]
                    refined_betas = refined_thetas[:, -10:]
                    
                    _, results_batch = collect_results_for_image_dapa(
                        refined_pose, refined_betas, pred_camera, refined_transl, batch, cam_focal_length, orig_img_width, orig_img_height,
                        smpl_type=args.smpl_model, kid_age=args.kid_age)

                    results_holder.update_results(batch['idx'].numpy(), results_batch)

                for idx_, res in enumerate(results_batch):
                    img_name = res['img_name']
                    joints = res['joints']
                    # rot_mtx = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
                    rot_mtx = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])[np.newaxis, :, :]
                    joints = joints @ rot_mtx - res['transl'][:, np.newaxis]

                    results_holder.update_scene(img_name, [joints[0, 11].tolist(), joints[0, 14].tolist()], cam_params, body_type)

        #################################################
        # Postprocessing: smoothing to remove jittering #
        #################################################
        if args.run_smplify:
            # Smoothing using OneEuroFilter.
            for track_id, pidxs in dataset.track_to_id.items():
                logger.info('Smoothing track {}'.format(track_id))
                pred_pose = np.stack([results_holder.results[i]['body_pose'][0] for i in pidxs])
                pred_orient = np.stack([results_holder.results[i]['global_orient'][0] for i in pidxs])
                pred_transl = np.stack([results_holder.results[i]['transl'][0] for i in pidxs])

                pose_filter = OneEuroFilter(np.zeros_like(pred_pose[0]), pred_pose[0], min_cutoff=0.004, beta=0.7)
                orient_filter = OneEuroFilter(np.zeros_like(pred_orient[0]), pred_orient[0], min_cutoff=0.004, beta=0.7)
                transl_filter = OneEuroFilter(np.zeros_like(pred_transl[0]), pred_transl[0], min_cutoff=0.004, beta=0.7)
                for i, pose in enumerate(pred_pose[1:]):
                    i += 1
                    pose = pose_filter(np.ones_like(pose) * i, pose)
                    orient = orient_filter(np.ones_like(pred_orient[i]) * i, pred_orient[i])
                    transl = transl_filter(np.ones_like(pred_transl[i]) * i, pred_transl[i])
                    pred_pose[i] = pose
                    pred_orient[i] = orient
                    pred_transl[i] = transl

                for i, idx in enumerate(pidxs):
                    results_holder.results_smoothed[idx] = results_holder.results[idx]
                    results_holder.results_smoothed[idx]['body_pose'] = pred_pose[i].reshape(1, -1)
                    results_holder.results_smoothed[idx]['global_orient'] = pred_orient[i].reshape(1, -1)
                    results_holder.results_smoothed[idx]['transl'] = pred_transl[i].reshape(1, -1)
        
        logger.info('Saving '+ os.path.join(out_folder, 'results.pt'))
        joblib.dump(results_holder, os.path.join(out_folder, 'results.pt'))

    else:
        logger.info('Loading '+ os.path.join(out_folder, 'results.pt'))
        results_holder = joblib.load(os.path.join(out_folder, 'results.pt'))

        scene_to_mean_floor = {}
        for scene_id, ((scene_start, scene_end), normal_vec) in dataset.ground_normals.items():
            break

    if args.ground_constraint and isinstance(normal_vec, torch.Tensor):
        normal_vec = normal_vec.cpu().numpy()
    if not args.ground_constraint:
        normal_vec = None
    
    post_fitting(dataset, results_holder, out_folder, cfg, device, args,
                camera_center, normal_vec, cam_params, images_folder,
                out_render_path)


def post_fitting(dataset, results_holder, out_folder, cfg, device, args,
                 camera_center, normal_vec, cam_params, images_folder,
                 out_render_path):
    # use the (mean) bottom of the baby or adult as the anchor for the ground plane
    if args.ground_anchor == 'child_bottom':
        anchor = np.concatenate([np.stack(ankles) for ankles in list(results_holder.infant_bottom.values())]).mean(0)
    elif args.ground_anchor == 'adult_bottom':
        anchor = np.concatenate([np.stack(ankles) for ankles in list(results_holder.adult_bottom.values())]).mean(0)

    if args.add_downstream:
        # returns a dictionary of results for each image
        filter_by_2dkp = args.pipeline == 1
        labels = get_downstream_labels(dataset, results_holder, filter_by_2dkp)

        with open(os.path.join(out_folder, 'labels.json'), 'w') as f:
            json.dump(labels, f)
    else:
        labels = None

    if args.render:
        render(
            dataset, results_holder, labels, images_folder, out_render_path, cfg, cam_params, 
            skip_if_no_infant=False, device=device, save_mesh=args.save_mesh,
            camera_center=camera_center, img_list=None, 
            fast_render=True, top_view=args.top_view, keep_criterion=args.keep,
            add_ground_plane=normal_vec is not None, anchor=anchor, ground_normal=normal_vec,
            renderer=args.renderer, smpl_type=args.smpl_model, kid_age=args.kid_age)

        if args.save_video:
            images_to_mp4(out_render_path, os.path.join(out_folder, 'video.mp4'), fps=args.fps)

        if args.save_gif:
            images_to_gif(out_render_path, os.path.join(out_folder, 'video.gif'), fps=args.fps)


if __name__ == '__main__':
    args = parse_config()
    main(args)
    