import os
import os.path as osp
import joblib
import pickle
import shutil
import re

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
from utils.geometry import batch_euler2matrix
from hps.dapa.dapa_utils import get_original, collect_results_for_image_dapa

# from utils.dapa_utils import collect_results_for_image_dapa, render_image_dapa, get_original

# from render_result import render_with_ground
# from utils.one_euro_filter import OneEuroFilter
# from utils.utils import render_image, render_image_group
# from pose_utils.runner import SMPLifyRunner, get_body_vertices_batched


def main(args):
    # convert video to images if necessary
    if args.video is not None and os.path.exists(args.video):
        from utils.video_utils import video_to_images # TODO (Jen)
        video_to_images(args.video, osp.join(osp.basename(args.video)))
        images_folder = osp.join(osp.basename(args.video), osp.splitext(osp.basename(args.video))[0])

    elif args.images is not None:
        images_folder = args.images
    
    out_folder = args.out_folder

    if os.path.exists(out_folder):
        logger.warning('Folder ' + out_folder + ' exists. Overwriting it...')
    else:
        os.makedirs(out_folder, exist_ok=True)

    shutil.copy(args.config, os.path.join(out_folder, 'config.yaml'))
    batch_size = args.batch_size
    device = 'cuda'
    
    # build dataset and save it to disk.
    dataset_path = os.path.join(out_folder, 'dataset.pt')
    if args.use_cached_dataset and os.path.exists(dataset_path):  
        logger.info('Loading dataset from ' + dataset_path) 
        dataset = joblib.load(dataset_path)
    else:  
        dataset = Dataset(
            images_folder, out_folder=out_folder, tracker_type=args.tracker_type, cfg=cfg
        )
        if args.ground_constraint:
            dataset.estimate_ground_plane_normal()
        joblib.dump(dataset, dataset_path)
        logger.info('Saved dataset.pt to folder: ' + out_folder)

    dataset.print_info()
    results_holder = Results()
    camera_center = dataset.camera_center
    orig_img_width, orig_img_height = camera_center[0] * 2, camera_center[1] * 2
    cam_focal_length = args.camera_focal
    focal_lengths = [cam_focal_length, cam_focal_length]

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # load pretrained DAPA model
    if args.hps == 'dapa':
        adult_model = hps.get_dapa_model(cfg.dapa_adult_model, cfg.smpl_mean_params).cuda().eval()
        infant_model = hps.get_dapa_model(cfg.dapa_child_model, cfg.smpl_mean_params).cuda().eval()
        create_body_pose = True
    elif args.hps == 'cliff':
        raise NotImplementedError('Cliff model not implemented yet.')
    else:
        create_body_pose = False  # if use vposer, then create_body_pose = False
        raise NotImplementedError('Unknown hps model: ' + args.hps)

    out_render_path = os.path.join(out_folder, 'render')
    if not os.path.exists(out_render_path):
        os.makedirs(out_render_path)

    if args.run_smplify:
        from postprocess.temporal_smplify import TemporalSMPLify
        smplify_runner = TemporalSMPLify(
            step_size=1e-2,
            num_iters=args.smplify_iters,
            focal_length=cam_focal_length,
            use_lbfgs=True,
            device=torch.device('cuda'),
            max_iter=20
        )
    else:
        smplify_runner = None
    
    # Begin main loop
    adult_tracks, infant_tracks = dataset.get_sorted_track_by_body_type()

    if args.ground_constraint:
        tracks_to_be_fitted = adult_tracks + adult_tracks + infant_tracks
        # Note: fit all tracks first, and then use the mean adult ankle location as floor to constrain the humans.
    else:
        tracks_to_be_fitted = adult_tracks + infant_tracks

    cam_pitch = cam_roll = 0.  # settings these would make render_rotmat an identity matrix.
    cam_vfov = 0.6  # this does not matter
    cam_rotmat = batch_euler2matrix(torch.tensor([[-cam_pitch, 0., cam_roll]]))[0].numpy()
    cam_params = np.array([cam_vfov, cam_pitch, cam_roll, cam_focal_length])   # Identity matrix

    for fit_id, track_id in enumerate(tracks_to_be_fitted):
        refine_with_ground = fit_id >= len(adult_tracks)  # iterate over tracks a second time and use the floor
        if fit_id == len(adult_tracks) and args.ground_constraint:
            logger.info('Fitting the rest of the tracks with ground constraint.')
            # compute ankles per scene
            scene_to_mean_floor = {}
            for scene_id, ((scene_start, scene_end), normal_vec) in dataset.ground_normals.items():
                normal_vec = normal_vec[[0,2,1]] 
                adult_ankles = [
                    results_holder.adult_bottom[dataset.img_names[frame_id]] 
                    for frame_id in range(scene_start, scene_end) if dataset.img_names[frame_id] in results_holder.adult_bottom
                ]
                if len(adult_ankles) == 0: 
                    scene_to_mean_floor[scene_id] = None
                    continue
                ankle_projections = np.concatenate(adult_ankles).flatten()
                mean = np.mean(ankle_projections)
                std = np.std(ankle_projections)
                mean_floor = np.mean(ankle_projections[np.abs(ankle_projections - mean) < std])  # filter out the outliers
                mean_floor = mean_floor * normal_vec  # location of the anchor ankle.
                scene_to_mean_floor[scene_id] = mean_floor
        
        pidxs = dataset.track_to_id[track_id]
        body_type = dataset.track_body_types[track_id][0]
        logger.info('Fitting track {} of length {}. Body type is {}...'.format(track_id, len(pidxs), body_type))

        for batch_start_i in range(0, len(pidxs), batch_size):
            batch_pidxs = pidxs[batch_start_i: batch_start_i+batch_size]
            cur_batch_size = len(batch_pidxs)
            batch = dataloader.collate_fn([dataset[i] for i in batch_pidxs])

            # image_id = int(os.path.splitext(batch['img_name'][0])[0].strip('frame_'))
            start_image_id = batch['idx'].item()

            ####################################
            # initialize pose, shape with DAPA #
            ####################################
            if args.ground_constraint:
                # load ground normal, and set cam_rotmat to be identity matrix
                for scene_id, (scene_rng, normal_vec) in dataset.ground_normals.items():
                    if start_image_id >= scene_rng[0] and start_image_id <= scene_rng[1]:
                        normal_vec = normal_vec[[0,2,1]]  # swap y and z
                        break
            
                if refine_with_ground:
                    logger.info(f'This scene is from frame {scene_rng[0]} to {scene_rng[1]}. Ground normal vector is {str(normal_vec)}.')
                    ground_y = scene_to_mean_floor[scene_id]
                    ground_y = torch.from_numpy(ground_y).to(device).float()
                    logger.info('Refine with floor at ' + str(ground_y))
                else:
                    ground_y = None

                normal_vec = torch.from_numpy(normal_vec).to(device).float()

            if args.hps == 'dapa':
                with torch.no_grad():
                    if body_type == 'infant':
                        pred_rotmat, pred_betas, pred_camera = infant_model(batch['norm_cropped_img'].to(device))
                    else:
                        pred_rotmat, pred_betas, pred_camera = adult_model(batch['norm_cropped_img'].to(device))

                init_betas = pred_betas
                pred_rotmat_hom = torch.cat(
                    [pred_rotmat.view(-1, 3, 3),
                    torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1).expand(cur_batch_size*24, -1, -1)
                    ], dim=-1)
                pred_pose = rotation_matrix_to_angle_axis(pred_rotmat_hom).contiguous().view(cur_batch_size, -1)
                pred_pose[torch.isnan(pred_pose)] = 0.0
                init_global_orient = pred_pose[:,:3]
                init_pose = pred_pose[:, 3:]

                # also pass in the transl, which is converted from pred_camera
                yhwx = batch['yhxw']
                y, h, x = yhwx[:,0], yhwx[:,1], yhwx[:,2]
                init_transl = []
                for i in range(cur_batch_size):
                    trans = get_original(pred_camera[i].detach().cpu().numpy(), x[i], y[i], h[i], cam_focal_length, 
                                         orig_img_width, orig_img_height)
                    init_transl.append(torch.from_numpy(trans))
                init_transl = torch.stack(init_transl).type(pred_camera.dtype).to(pred_camera.device)

            else:
                # TODO: (option to use phalp as initialization)
                init_betas = init_global_orient = init_pose = init_transl = None
            
            ##############################################
            # Collect results and optionally run SMPLify #
            ##############################################
            if not args.run_smplify:
                # parse results
                _, results_batch = collect_results_for_image_dapa(
                    pred_pose, pred_betas, pred_camera, None, batch, cam_focal_length, orig_img_width, orig_img_height)
                # update results in this batch
                results_holder.update_results(batch['idx'].numpy(), results_batch)
            else:
                model_type = 'smil' if body_type == 'infant' else 'smpl'

                smplify_results, reproj_loss = smplify_runner(model_type, init_global_orient, init_pose, init_betas, init_transl, 
                                                              camera_center, batch['keypoints'], ground_y, normal_vec)
                refined_thetas = smplify_results['theta']
                refined_transl = refined_thetas[:, :3]
                refined_pose = refined_thetas[:, 3:-10]
                refined_betas = refined_thetas[:, -10:]
                
                _, results_batch = collect_results_for_image_dapa(
                    refined_pose, refined_betas, pred_camera, refined_transl, batch, cam_focal_length, orig_img_width, orig_img_height)

                results_holder.update_results(batch['idx'].numpy(), results_batch)

                for idx_, res in enumerate(results_batch):
                    img_name = res['img_name']
                    joints = res['joints']
                    # project the left/right ankles onto the normal_vec (unit-norm)
                    left_ankle_projection = np.dot(joints[0, 11], normal_vec.cpu().numpy())
                    right_ankle_projection = np.dot(joints[0, 14], normal_vec.cpu().numpy())
                    results_holder.update_scene(img_name, [left_ankle_projection, right_ankle_projection], cam_params, body_type)
       

                # TODO (Jen): implement later.
                #################################################
                # fill in empty kp2d for PHALP ghost detections #
                #################################################
                # is_ghost = batch['is_ghost'].numpy()
                # ghost_idxs = np.where(is_ghost)[0]

                # for ghost_idx in ghost_idxs:
                #     if ghost_idx == 0:
                #         batch['keypoints'][0] = batch['keypoints'][np.where(~is_ghost)[0][0]]
                #         batch['yhxw'][0] = batch['yhxw'][np.where(~is_ghost)[0][0]]
                    
                #     batch['keypoints'][ghost_idx] = batch['keypoints'][ghost_idx-1]
                #     batch['yhxw'][ghost_idx] = batch['yhxw'][ghost_idx-1]
             
                # # Fit the body models for this batch
                # if body_type == 'infant':
                #     bm = hps.init_body_model(cfg.smil_model_path, batch_size=cur_batch_size, create_body_pose=create_body_pose)
                #     bm_single = hps.init_body_model(cfg.smil_model_path, batch_size=1, create_body_pose=create_body_pose)
                # else:
                #     bm = hps.init_body_model(cfg.smpl_model_path, batch_size=cur_batch_size, create_body_pose=create_body_pose)
                #     bm_single = hps.init_body_model(cfg.smpl_model_path, batch_size=1, create_body_pose=create_body_pose)
                #     init_pose = init_pose[:, :63]
    
                
                # verts = faces = qp_sdfs = vmin = vmax = cam_R = ground_y = None
                # grid_dim = 8
                # results = smplify_runner.fit(batch, bm, args.smplify_iters, model_type, 
                #             init_betas, init_global_orient, init_pose, init_transl,
                #             verts, faces, qp_sdfs, vmin, vmax, grid_dim, cam_R, ground_y, normal_vec, 
                #             spec_focal=cam_focal_length, spec_camrot=cam_rotmat.cuda())
                # results_holder.update_results(batch['idx'].numpy(), results)

                # Humans in this batch come from different images. Clean up results by 
                # mapping each image to the fittings from that image.
                # body_verts = get_body_vertices_batched(bm_single, results, convert=False, transl_body=False)
                # body_verts_with_transl = get_body_vertices_batched(bm_single, results, convert=False, transl_body=True)
                
                # results_by_img_name = dict()
                # for img_name in set(batch['img_name']):
                #     idx = np.where(np.array(batch['img_name'])==img_name)[0]
                #     # assert len(idx) == 1, batch['img_name']
                #     results_by_img_name[img_name] = [
                #         [body_verts[i] for i in idx],
                #         [body_verts_with_transl[i] for i in idx],
                #         [bm_single.faces for i in idx],
                #         [batch['keypoints'][i][:25].numpy() for i in idx],
                #         [init_transl[i] for i in idx],
                #         [results[i]['transl'] for i in idx],
                #         [results[i]['joints'][0, :25] for i in idx],
                #     ]
                
                # for idx_, (img_name, res) in enumerate(results_by_img_name.items()):

                #     vertices, vertices_with_transl, faces, keypoints_2d, transl, joints = res[0][0], res[1][0], res[2][0], res[3][0], res[5][0].flatten(), res[6][0]
                
                #     # project the left/right ankles onto the normal_vec (unit-norm)
                #     left_ankle_projection = np.dot(joints[11], normal_vec.cpu().numpy())
                #     right_ankle_projection = np.dot(joints[14], normal_vec.cpu().numpy())
                #     results_holder.update_scene(img_name, [left_ankle_projection, right_ankle_projection], cam_params, body_type)
       
                #     if refine_with_ground:
                #         mesh_color = 'green' if body_type == 'adult' else 'blue'
                #         if keypoints_2d[:,2].sum() > 9: 
                #             mesh_color = 'dark_'+mesh_color
                #         elif keypoints_2d[:,2].sum() < 5:
                #             mesh_color = 'light_'+mesh_color
                #         save_filename = os.path.join(out_render_path, img_name)
                        
                #         if idx_ == 0:  # no need to render all frames. this will be done at the end by function render_with_ground
                #             render_image_group(
                #                 os.path.join(images_folder, img_name), transl, vertices, render_rotmat, 
                #                 focal_lengths, camera_center,
                #                 faces=faces, cam_params=cam_params, keypoints_2d=keypoints_2d, save_filename=save_filename,
                #                 ground_y=ground_y,
                #                 mesh_color=mesh_color, fast_render=True)

    #################################################
    # Postprocessing: smoothing to remove jittering #
    #################################################
    # if args.run_smplify:
    #     # Smoothing using OneEuroFilter (as in SPIN).
    #     for track_id, pidxs in dataset.track_to_id.items():
    #         logger.info('Smoothing track {}'.format(track_id))
    #         pred_pose = np.stack([results_holder.results[i]['body_pose'][0] for i in pidxs])
    #         pred_orient = np.stack([results_holder.results[i]['global_orient'][0] for i in pidxs])
    #         pred_transl = np.stack([results_holder.results[i]['transl'][0] for i in pidxs])

    #         pose_filter = OneEuroFilter(np.zeros_like(pred_pose[0]), pred_pose[0], min_cutoff=0.004, beta=0.7)
    #         orient_filter = OneEuroFilter(np.zeros_like(pred_orient[0]), pred_orient[0], min_cutoff=0.004, beta=0.7)
    #         transl_filter = OneEuroFilter(np.zeros_like(pred_transl[0]), pred_transl[0], min_cutoff=0.004, beta=0.7)
    #         for i, pose in enumerate(pred_pose[1:]):
    #             i += 1
    #             pose = pose_filter(np.ones_like(pose) * i, pose)
    #             orient = orient_filter(np.ones_like(pred_orient[i]) * i, pred_orient[i])
    #             transl = transl_filter(np.ones_like(pred_transl[i]) * i, pred_transl[i])
    #             pred_pose[i] = pose
    #             pred_orient[i] = orient
    #             pred_transl[i] = transl

    #         for i, idx in enumerate(pidxs):
    #             results_holder.results_smoothed[idx] = results_holder.results[idx]
    #             results_holder.results_smoothed[idx]['body_pose'] = pred_pose[i].reshape(1, -1)
    #             results_holder.results_smoothed[idx]['global_orient'] = pred_orient[i].reshape(1, -1)
    #             results_holder.results_smoothed[idx]['transl'] = pred_transl[i].reshape(1, -1)
    
    logger.info('Saving '+ os.path.join(out_folder, 'results.pt'))
    joblib.dump(results_holder, os.path.join(out_folder, 'results.pt'))

    if args.render:
        from visualization.pyrender import render
        render(
            dataset, results_holder, images_folder, out_render_path, cfg, cam_params, 
            skip_if_no_infant=False, device=device, save_mesh=args.save_mesh,
            camera_center=camera_center, img_list=None, 
            add_ground_plane=True, fast_render=True, top_view=args.top_view)

        if args.save_video:
            cmd = [
                'ffmpeg',
                '-y',
                '-framerate', str(args.fps),
                '-i', '{}/frame_%08d.jpg'.format(out_render_path),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'veryslow',
                '{}/video.mp4'.format(out_folder)
            ]
            cmd = ' '.join(cmd)
            print(cmd)
            os.system(cmd)


if __name__ == '__main__':
    args = parse_config()
    main(args)
    